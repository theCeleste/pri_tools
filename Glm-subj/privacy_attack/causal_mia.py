#!/usr/bin/env python3
"""Indirect membership inference using ChatGLM causal token likelihoods."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import zlib

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    load_split,
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
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets-per-split", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-k-percent", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    if not args.adapter and not args.steering:
        parser.error("at least one of --adapter or --steering is required")
    if args.targets_per_split < 1 or args.max_length < 2:
        parser.error("target count must be positive and max length at least 2")
    if not 0 < args.min_k_percent <= 1:
        parser.error("min-k-percent must be in (0,1]")
    if args.steering_multiplier is not None and args.steering_multiplier == 0:
        parser.error("steering-multiplier cannot be zero")


def roc_metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    fpr, tpr, _ = roc_curve(labels_array, scores_array)

    def tpr_at(limit):
        eligible = tpr[fpr <= limit]
        return float(eligible.max()) if len(eligible) else 0.0

    return {"auc": float(roc_auc_score(labels_array, scores_array)),
            "tpr_at_1pct_fpr": tpr_at(0.01), "tpr_at_0_1pct_fpr": tpr_at(0.001)}


def run(args):
    from contextlib import nullcontext
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Causal membership inference requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
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

    @torch.inference_mode()
    def score(text):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with target_context():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        token_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:]
        selected = token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        valid = attention_mask[:, 1:].bool()[0]
        selected = selected[valid]
        if not len(selected):
            raise ValueError("Text has no scoreable tokens")
        k = max(1, math.ceil(len(selected) * args.min_k_percent))
        mean_logprob = float(selected.mean().cpu())
        total_logprob = float(selected.sum().cpu())
        min_k_logprob = float(torch.topk(selected, k, largest=False).values.mean().cpu())
        compressed = max(len(zlib.compress(text.encode("utf-8"))), 1)
        return {"mean_logprob": mean_logprob, "min_k_logprob": min_k_logprob,
                "zlib_normalized_logprob": total_logprob / compressed,
                "tokens": int(len(selected)), "zlib_bytes": compressed}

    records = []
    for split_index, split in enumerate(("train", "test")):
        rows = load_split(args.data_dir, split)
        indexes = list(range(len(rows)))
        random.Random(args.seed + split_index).shuffle(indexes)
        for index in indexes[:args.targets_per_split]:
            values = score(rows[index]["text"])
            records.append({"split": split, "member": int(split == "train"),
                            "dataset_index": index, "text": rows[index]["text"], **values})

    labels = [row["member"] for row in records]
    metrics = {name: roc_metrics(labels, [row[name] for row in records])
               for name in ("mean_logprob", "min_k_logprob", "zlib_normalized_logprob")}
    best_name = max(metrics, key=lambda name: metrics[name]["auc"])
    target_type = (
        f"causal_lm_adapter+{steering_config['method']}"
        if args.adapter and steering_config
        else "causal_lm_adapter"
        if args.adapter
        else steering_config["method"]
    )
    summary = {"attack": "causal_loss_membership_inference",
               "target_type": target_type,
               "configuration": vars(args),
               "metrics": metrics, "best_metric": best_name, "best": metrics[best_name],
               "members": sum(labels), "nonmembers": len(labels) - sum(labels)}
    output_dir = require_empty_output(args.output_dir)
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
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
