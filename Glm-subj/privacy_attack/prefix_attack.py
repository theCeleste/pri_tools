#!/usr/bin/env python3
"""Known-prefix extraction for a ChatGLM3 adapter, steering target, or both."""

from __future__ import annotations

import argparse
import json
import random

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    fuzzy_text_metrics,
    load_split,
    mean,
    require_empty_output,
    validate_adapter,
    validate_chatglm_base,
    write_json,
    write_jsonl,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter", help="Optional CAUSAL_LM PEFT adapter directory")
    parser.add_argument("--steering", help="Optional mean/private steering directory")
    parser.add_argument("--steering-multiplier", type=float)
    parser.add_argument("--prompt-mode", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets-per-split", type=int, default=100)
    parser.add_argument("--samples-per-prefix", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=2)
    parser.add_argument("--min-suffix-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature-start", type=float, default=1.20)
    parser.add_argument("--temperature-end", type=float, default=1.00)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.65)
    parser.add_argument("--minimum-fuzzy-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    if not args.adapter and not args.steering:
        parser.error("at least one of --adapter or --steering is required")
    positive = ("targets_per_split", "samples_per_prefix", "prefix_tokens", "min_suffix_tokens",
                "max_new_tokens", "top_k", "minimum_fuzzy_tokens")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("sample and token counts must be positive")
    if args.temperature_start <= 0 or args.temperature_end <= 0 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling parameters")
    if not 0 <= args.fuzzy_threshold <= 1:
        parser.error("fuzzy-threshold must be in [0,1]")
    if args.steering_multiplier is not None and args.steering_multiplier == 0:
        parser.error("steering-multiplier cannot be zero")


def select_targets(rows, tokenizer, args, seed):
    indexes = list(range(len(rows)))
    random.Random(seed).shuffle(indexes)
    selected = []
    for index in indexes:
        explicit_prefix = rows[index].get("audit_prefix")
        explicit_reference = rows[index].get("audit_reference")
        if explicit_prefix is not None or explicit_reference is not None:
            if not isinstance(explicit_prefix, str) or not explicit_prefix.strip():
                raise ValueError(f"Row {index} has an invalid audit_prefix")
            if not isinstance(explicit_reference, str) or not explicit_reference.strip():
                raise ValueError(f"Row {index} has an invalid audit_reference")
            prefix_ids = tokenizer(explicit_prefix.strip(), add_special_tokens=False)["input_ids"]
            reference_ids = tokenizer(explicit_reference.strip(), add_special_tokens=False)["input_ids"]
            if prefix_ids and isinstance(prefix_ids[0], list):
                prefix_ids = prefix_ids[0]
            if reference_ids and isinstance(reference_ids[0], list):
                reference_ids = reference_ids[0]
            if len(reference_ids) < args.min_suffix_tokens:
                continue
            selected.append({
                "dataset_index": index,
                "prefix_ids": list(prefix_ids),
                "reference_ids": list(reference_ids[:args.max_new_tokens]),
                "explicit_prefix": explicit_prefix.strip(),
            })
            if len(selected) == args.targets_per_split:
                break
            continue
        ids = tokenizer(rows[index]["text"], add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) < args.prefix_tokens + args.min_suffix_tokens:
            continue
        selected.append({"dataset_index": index, "prefix_ids": list(ids[:args.prefix_tokens]),
                         "reference_ids": list(ids[args.prefix_tokens:args.prefix_tokens + args.max_new_tokens])})
        if len(selected) == args.targets_per_split:
            break
    if len(selected) < args.targets_per_split:
        raise ValueError(f"Only {len(selected)} rows are long enough; requested {args.targets_per_split}")
    return selected


def summarize(records):
    attempts_by_target = {}
    for record in records:
        key = (record["split"], record["dataset_index"])
        attempts_by_target.setdefault(key, []).append(record)

    # Report target-level success when any sample extracts that target. Prefer a
    # successful sample as the representative used by the per-target means.
    best = {
        key: max(
            attempts,
            key=lambda record: (
                record["fuzzy_success"],
                record["exact_success"],
                record["normalized_exact_success"],
                record["continuous_token_match"],
                record["fuzzy_score"],
            ),
        )
        for key, attempts in attempts_by_target.items()
    }
    values = list(best.values())
    target_exact_hits = sum(record["exact_success"] for record in values)
    target_normalized_exact_hits = sum(record["normalized_exact_success"] for record in values)
    target_fuzzy_hits = sum(record["fuzzy_success"] for record in values)
    summary = {
        "targets": len(values),
        "generations": len(records),
        "target_exact_hits": target_exact_hits,
        "target_normalized_exact_hits": target_normalized_exact_hits,
        "target_fuzzy_hits": target_fuzzy_hits,
        "generation_exact_hits": sum(record["exact_success"] for record in records),
        "generation_normalized_exact_hits": sum(
            record["normalized_exact_success"] for record in records
        ),
        "generation_fuzzy_hits": sum(record["fuzzy_success"] for record in records),
        "exact_success_rate": target_exact_hits / len(values) if values else 0.0,
        "normalized_exact_success_rate": (
            target_normalized_exact_hits / len(values) if values else 0.0
        ),
        "fuzzy_success_rate": target_fuzzy_hits / len(values) if values else 0.0,
        "mean_fuzzy_score": mean(values, "fuzzy_score"),
        "mean_rouge_l": mean(values, "rouge_l"),
        "mean_token_f1": mean(values, "token_f1"),
        "mean_char_similarity": mean(values, "char_similarity"),
        "mean_prefix_match_tokens": mean(values, "prefix_match_tokens"),
        "mean_continuous_token_match": mean(values, "continuous_token_match"),
        "mean_token_accuracy": mean(values, "token_accuracy"),
    }
    for threshold in (4, 8, 16, 32):
        eligible = [
            attempts for attempts in attempts_by_target.values()
            if attempts[0]["reference_token_ids"] >= threshold
        ]
        target_hits = sum(
            any(record["continuous_token_match"] >= threshold for record in attempts)
            for attempts in eligible
        )
        summary[f"target_hits_at_{threshold}_tokens"] = target_hits
        summary[f"eligible_targets_at_{threshold}_tokens"] = len(eligible)
        summary[f"hit_rate_at_{threshold}_tokens"] = (
            target_hits / len(eligible)
            if eligible else 0.0
        )
    return summary


def run(args):
    from contextlib import nullcontext
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Prefix extraction requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    steering_directions = None
    steering_config = None
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    if args.steering:
        from .steering import load_steering
        steering_config, steering_directions = load_steering(args.steering)
    model = model.to(device).eval()

    def target_context():
        if steering_directions is None:
            return nullcontext()
        from .steering import apply_steering
        multiplier = (args.steering_multiplier if args.steering_multiplier is not None
                      else float(steering_config["inference_multiplier"]))
        return apply_steering(model, steering_directions, multiplier)

    class TemperatureDecay(LogitsProcessor):
        def __init__(self, prompt_length, generation_limit):
            self.prompt_length = prompt_length
            self.generation_limit = generation_limit

        def __call__(self, input_ids, scores):
            generated = max(0, input_ids.shape[-1] - self.prompt_length)
            progress = min(generated / max(self.generation_limit - 1, 1), 1.0)
            temperature = args.temperature_start + progress * (args.temperature_end - args.temperature_start)
            return scores / temperature

    def strip_stop(ids):
        stops = {value for value in (tokenizer.eos_token_id, tokenizer.pad_token_id) if value is not None}
        result = []
        for token_id in ids:
            if int(token_id) in stops:
                break
            result.append(int(token_id))
        return result

    records = []
    for split_index, split in enumerate(("train", "test")):
        targets = select_targets(load_split(args.data_dir, split), tokenizer, args, args.seed + split_index)
        for target_index, target in enumerate(targets):
            prefix_text = target.get("explicit_prefix") or tokenizer.decode(
                target["prefix_ids"], skip_special_tokens=True
            ).strip()
            # A causal adapter is trained directly on raw Subj text, so its
            # prefix must retain that training format even when steering hooks
            # are applied on top. Steering-only targets keep the reference
            # implementation's instruction-style prompt.
            prompt_mode = (
                "chat"
                if args.prompt_mode == "auto" and args.steering and not args.adapter
                else args.prompt_mode
            )
            if prompt_mode == "auto":
                prompt_mode = "raw"
            if prompt_mode == "chat":
                query = (
                    "Continue this exact Subj sentence. Output only the continuation, without explanation.\n"
                    f"Sentence prefix: {prefix_text}"
                )
                prompt_ids = tokenizer.build_chat_input(query)["input_ids"][0].tolist()
            else:
                prompt_ids = tokenizer.build_inputs_with_special_tokens(target["prefix_ids"])
            encoded = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generation_limit = min(args.max_new_tokens, len(target["reference_ids"]) + 16)
            for sample_index in range(args.samples_per_prefix):
                sample_seed = args.seed + split_index * 1_000_000 + target_index * args.samples_per_prefix + sample_index
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                with torch.inference_mode(), target_context():
                    output = model.generate(
                        input_ids=encoded, attention_mask=torch.ones_like(encoded),
                        max_new_tokens=generation_limit, do_sample=True,
                        top_p=args.top_p, top_k=args.top_k,
                        logits_processor=[TemperatureDecay(encoded.shape[-1], generation_limit)],
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = strip_stop(output[0, encoded.shape[-1]:].tolist())
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                reference = tokenizer.decode(target["reference_ids"], skip_special_tokens=True).strip()
                scored_ids = generated_ids[:len(target["reference_ids"])]
                scored_generated = tokenizer.decode(scored_ids, skip_special_tokens=True).strip()
                metrics = fuzzy_text_metrics(scored_generated, reference)
                token_exact = generated_ids[:len(target["reference_ids"])] == target["reference_ids"]
                continuous_token_match = 0
                for generated_id, reference_id in zip(generated_ids, target["reference_ids"]):
                    if generated_id != reference_id:
                        break
                    continuous_token_match += 1
                fuzzy_success = token_exact or metrics["normalized_exact"] or (
                    continuous_token_match >= args.minimum_fuzzy_tokens
                    and metrics["fuzzy_score"] >= args.fuzzy_threshold
                )
                records.append({
                    "split": split, "dataset_index": target["dataset_index"],
                    "sample_index": sample_index, "seed": sample_seed,
                    "prefix": prefix_text, "reference": reference,
                    "generated": generated, "scored_generated": scored_generated, **metrics,
                    "reference_token_ids": len(target["reference_ids"]),
                    "continuous_token_match": continuous_token_match,
                    "exact_success": int(token_exact),
                    "normalized_exact_success": int(metrics["normalized_exact"]),
                    "fuzzy_success": int(fuzzy_success),
                })

    output_dir = require_empty_output(args.output_dir)
    write_jsonl(output_dir / "records.jsonl", records)
    by_split = {split: summarize([row for row in records if row["split"] == split])
                for split in ("train", "test")}
    comparable = ("fuzzy_success_rate", "mean_fuzzy_score", "mean_rouge_l", "mean_token_f1",
                  "mean_char_similarity", "mean_prefix_match_tokens", "mean_continuous_token_match",
                  "mean_token_accuracy")
    summary = {
        "attack": "known_prefix_causal_extraction",
        "target_type": (
            f"causal_lm_adapter+{steering_config['method']}"
            if args.adapter and steering_config
            else "causal_lm_adapter"
            if args.adapter
            else steering_config["method"]
        ),
        "configuration": vars(args),
        "train": by_split["train"],
        "holdout": by_split["test"],
        "train_minus_holdout": {key: by_split["train"][key] - by_split["test"][key]
                                for key in comparable},
    }
    write_json(output_dir / "summary.json", summary)
    ranked = sorted(records, key=lambda row: (row["fuzzy_success"], row["fuzzy_score"]), reverse=True)
    with (output_dir / "top_matches.txt").open("w", encoding="utf-8") as handle:
        for row in ranked[:100]:
            handle.write(f"[{row['split']}] fuzzy={row['fuzzy_score']:.4f} success={row['fuzzy_success']}\n"
                         f"PREFIX: {row['prefix']}\nREFERENCE: {row['reference']}\n"
                         f"GENERATED: {row['generated']}\nSCORED: {row['scored_generated']}\n\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    validate_chatglm_base(args.base_model)
    if args.adapter:
        validate_adapter(args.adapter)
    if args.steering:
        from .steering import validate_steering
        validate_steering(args.steering)
    for split in ("train", "test"):
        load_split(args.data_dir, split)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
