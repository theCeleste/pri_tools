#!/usr/bin/env python3
"""Generate a Subj training set by noisy aggregation of private few-shot groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    load_split,
    require_empty_output,
    validate_chatglm_base,
    write_json,
    write_jsonl,
)


LABEL_NAMES = {0: "objective", 1: "subjective"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", default="outputs/data/example_protected")
    parser.add_argument("--num-synthetic", type=int, default=512)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--examples-per-group", type=int, default=4)
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    positive = ("num_synthetic", "num_groups", "examples_per_group", "candidate_top_k",
                "max_prompt_tokens", "max_new_tokens", "min_new_tokens")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("counts and token limits must be positive")
    if args.min_new_tokens > args.max_new_tokens:
        parser.error("min-new-tokens cannot exceed max-new-tokens")
    if not math.isfinite(args.sigma) or args.sigma < 0 or args.temperature <= 0:
        parser.error("sigma must be nonnegative and temperature positive")


def make_groups(rows, private_pool, anchor, args, rng):
    label = int(rows[anchor]["label"])
    candidates = [index for index in private_pool
                  if index != anchor and int(rows[index]["label"]) == label]
    needed = args.num_groups * args.examples_per_group - 1
    if len(candidates) < needed:
        raise ValueError(f"Not enough label-{label} members for disjoint private groups")
    selected = rng.sample(candidates, needed) + [anchor]
    rng.shuffle(selected)
    return [selected[start:start + args.examples_per_group]
            for start in range(0, len(selected), args.examples_per_group)]


def prompt_for(rows, indexes, label):
    examples = "\n".join(f"- {rows[index]['text']}" for index in indexes)
    return (
        "Generate one short English movie-review sentence for the Subj dataset.\n"
        f"Required label: {LABEL_NAMES[label]}.\n"
        "Infer style from the examples without copying one verbatim.\n"
        f"Examples:\n{examples}\nNew sentence:"
    )


def fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    return digest.hexdigest()


def run(args):
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Example aggregation requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = require_empty_output(args.output_dir)
    train = load_split(args.data_dir, "train")
    validation = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    if args.num_synthetic > len(train):
        raise ValueError("num-synthetic exceeds the raw training-set size")

    anchors = list(range(len(train)))
    random.Random(args.seed).shuffle(anchors)
    anchors = anchors[:args.num_synthetic]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()

    newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    stop_ids = {int(tokenizer.eos_token_id)}
    if len(newline_ids) == 1:
        stop_ids.add(int(newline_ids[0]))
    noise_std = math.sqrt(2.0) * args.sigma / args.num_groups
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 10_000)

    def generate(prompts):
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=args.max_prompt_tokens)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        position_ids = model.get_position_ids(input_ids, device=device)
        past = None
        first = True
        generated = []
        for token_index in range(args.max_new_tokens):
            prepared = model.prepare_inputs_for_generation(
                input_ids, past_key_values=past, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=True, is_first_forward=first,
            )
            with torch.inference_mode():
                outputs = model(**prepared)
            probabilities = torch.softmax(outputs.logits[:, -1, :].float() / args.temperature, dim=-1)
            values, ids = torch.topk(probabilities, args.candidate_top_k, dim=-1)
            values = values / values.sum(dim=-1, keepdim=True)
            aggregate = torch.zeros(probabilities.shape[-1], dtype=torch.float32, device=device)
            aggregate.scatter_add_(0, ids.reshape(-1), values.reshape(-1))
            aggregate /= args.num_groups
            candidates = torch.unique(ids)
            scores = aggregate[candidates]
            if noise_std:
                scores += torch.randn(scores.shape, device=device, generator=noise_rng) * noise_std
            if token_index < args.min_new_tokens:
                for stop_id in stop_ids:
                    scores[candidates == stop_id] = -torch.inf
            next_id = int(candidates[int(scores.argmax())])
            if next_id in stop_ids:
                break
            generated.append(next_id)
            past = outputs.past_key_values
            input_ids = torch.cat([input_ids, input_ids.new_full((input_ids.shape[0], 1), next_id)], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            position_ids = torch.cat([position_ids, position_ids[..., -1:] + 1], dim=-1)
            first = False
        text = tokenizer.decode(generated, skip_special_tokens=True).strip().strip('"').strip("'")
        return text.split("\n", 1)[0].strip()

    synthetic = []
    originals = []
    provenance = []
    started = time.time()
    for synthetic_index, anchor in enumerate(tqdm(anchors, desc="example aggregation")):
        rng = random.Random(args.seed * 1_000_003 + synthetic_index)
        groups = make_groups(train, anchors, anchor, args, rng)
        label = int(train[anchor]["label"])
        text = generate([prompt_for(train, group, label) for group in groups])
        if not text:
            raise RuntimeError(f"Empty synthetic output at index {synthetic_index}")
        sources = [index for group in groups for index in group]
        synthetic.append({"sentence": text, "label": label, "_anchor_source_index": anchor,
                          "_private_source_indices": sources})
        originals.append({**train[anchor], "_source_index": anchor})
        provenance.append({"synthetic_index": synthetic_index, "anchor_source_index": anchor,
                           "private_source_indices": sources})

    for split, rows in (("train", synthetic), ("validation", validation), ("test", test)):
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    original_dir = output_dir / "original_data"
    for split, rows in (("train", originals), ("validation", validation), ("test", test)):
        write_jsonl(original_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "provenance.jsonl", provenance)
    manifest = {
        "dataset": "subj",
        "method": "noisy_private_example_aggregation",
        "formal_dp_claim": False,
        "privacy_note": "Adaptive token composition and source-record reuse need a separate accountant.",
        "arguments": vars(args),
        "noise_std_per_token": noise_std,
        "source_fingerprint": fingerprint([train[index] for index in anchors]),
        "synthetic_fingerprint": fingerprint(synthetic),
        "rows": {"train": len(synthetic), "validation": len(validation), "test": len(test)},
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    validate_chatglm_base(args.base_model)
    train = load_split(args.data_dir, "train")
    for split in ("validation", "test"):
        load_split(args.data_dir, split)
    if args.num_synthetic > len(train):
        parser.error("num-synthetic exceeds train size")
    if args.dry_run:
        print(json.dumps({**vars(args), "noise_std_per_token": math.sqrt(2) * args.sigma / args.num_groups},
                         ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
