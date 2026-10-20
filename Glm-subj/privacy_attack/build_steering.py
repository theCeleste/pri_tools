#!/usr/bin/env python3
"""Build the original mean/private example-steering methods for ChatGLM3/Subj."""

from __future__ import annotations

import argparse
import json
import random
import time

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    load_split,
    require_empty_output,
    save_audit_splits,
    validate_chatglm_base,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", default="outputs/models/example")
    parser.add_argument("--variants", nargs="+", choices=["mean", "private"],
                        default=["mean", "private"])
    parser.add_argument("--construction-samples", type=int, default=1000)
    parser.add_argument("--layers", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument("--read-token-index", type=int, default=-2)
    parser.add_argument("--clip", type=float, default=18.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.005)
    parser.add_argument("--multiplier", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser, train_size=None):
    if args.construction_samples < 1 or not args.layers or not args.variants:
        parser.error("construction-samples and layers must be non-empty")
    if args.clip <= 0 or args.noise_multiplier < 0 or args.multiplier == 0:
        parser.error("invalid steering parameters")
    if len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        parser.error("layers must be unique nonnegative indexes")
    if train_size is not None and args.construction_samples > train_size:
        parser.error("construction-samples exceeds the Subj train split")


def subj_contrast(row):
    if "steering_positive" in row or "steering_negative" in row:
        positive = row.get("steering_positive")
        negative = row.get("steering_negative")
        if not isinstance(positive, str) or not positive.strip():
            raise ValueError("steering_positive must be a non-empty string")
        if not isinstance(negative, str) or not negative.strip():
            raise ValueError("steering_negative must be a non-empty string")
        return positive.strip(), negative.strip()
    sentence = row["text"]
    question = (
        "Is the following sentence subjective or objective?\n\n"
        f'Sentence: "{sentence}"\n\n'
        "Options:\n(A) The sentence is subjective.\n(B) The sentence is objective."
    )
    subjective = "(A) The sentence is subjective."
    objective = "(B) The sentence is objective."
    matching = subjective if int(row["label"]) == 1 else objective
    nonmatching = objective if int(row["label"]) == 1 else subjective
    prefix = f"<|user|>\n{question}<|assistant|>\n"
    return prefix + matching, prefix + nonmatching


def run(args):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from tqdm import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Steering construction requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = require_empty_output(args.output_dir)
    train = [dict(row, _source_index=index) for index, row in enumerate(load_split(args.data_dir, "train"))]
    validation = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    random.Random(args.seed).shuffle(train)
    selected = train[:args.construction_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    if hasattr(tokenizer, "encode_special_tokens"):
        tokenizer.encode_special_tokens = True
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    if max(args.layers) >= len(model.transformer.encoder.layers):
        raise ValueError(f"Layer index exceeds model depth {len(model.transformer.encoder.layers)}")

    difference_sums = {}
    clipped_sums = {}
    max_norms = {layer: 0.0 for layer in args.layers}

    @torch.inference_mode()
    def extract(text):
        captured = {}
        handles = []
        for layer_index in args.layers:
            layer = model.transformer.encoder.layers[layer_index]

            def make_hook(index):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    token_index = args.read_token_index if args.read_token_index >= 0 else hidden.shape[0] + args.read_token_index
                    if not 0 <= token_index < hidden.shape[0]:
                        raise ValueError(f"Prompt too short for read-token-index={args.read_token_index}")
                    captured[index] = hidden[token_index, 0].detach().float().cpu().clone()
                return hook

            handles.append(layer.register_forward_hook(make_hook(layer_index)))
        try:
            encoded = tokenizer(text, return_tensors="pt")
            model(**{key: value.to(device) for key, value in encoded.items()}, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(args.layers):
            raise RuntimeError("Failed to capture all requested ChatGLM layers")
        return captured

    started = time.time()
    for row in tqdm(selected, desc="building mean/private steering"):
        positive_text, negative_text = subj_contrast(row)
        positive = extract(positive_text)
        negative = extract(negative_text)
        for layer_index in args.layers:
            difference = positive[layer_index] - negative[layer_index]
            difference_sums.setdefault(layer_index, torch.zeros_like(difference)).add_(difference)
            norm = float(difference.norm())
            max_norms[layer_index] = max(max_norms[layer_index], norm)
            clipped_sums.setdefault(layer_index, torch.zeros_like(difference)).add_(
                difference, alpha=min(1.0, args.clip / max(norm, 1e-12))
            )

    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    mean_directions = {}
    private_directions = {}
    layer_stats = {}
    for layer_index in args.layers:
        mean_direction = (difference_sums[layer_index] / len(selected)) / max(max_norms[layer_index], 1e-12)
        private_signal = clipped_sums[layer_index] / (len(selected) * args.clip)
        noise = torch.randn(private_signal.shape, generator=noise_generator) * args.noise_multiplier
        mean_directions[f"layer_{layer_index}"] = mean_direction.contiguous()
        private_directions[f"layer_{layer_index}"] = (private_signal + noise).contiguous()
        layer_stats[str(layer_index)] = {
            "mean_norm": float(mean_direction.norm()),
            "private_signal_norm": float(private_signal.norm()),
            "private_noise_norm": float(noise.norm()),
            "maximum_record_difference_norm": max_norms[layer_index],
        }

    common = {
        "model_type": "chatglm",
        "base_model": str(args.base_model),
        "layers": args.layers,
        "construction_samples": len(selected),
        "read_token_index": args.read_token_index,
        "inference_multiplier": args.multiplier,
        "seed": args.seed,
        "causal_training": False,
        "source_method": "example steering",
        "layer_stats": layer_stats,
    }
    variants = (
        ("mean", "mean_steering", mean_directions),
        ("private", "private_steering", private_directions),
    )
    for name, method, directions in variants:
        if name not in args.variants:
            continue
        directory = root / name
        directory.mkdir()
        save_file(directions, str(directory / "steering.safetensors"))
        metadata = {**common, "method": method, "protected": name == "private"}
        if name == "private":
            metadata.update({
                "clip": args.clip,
                "noise_multiplier": args.noise_multiplier,
                "formal_dp_claim": False,
                "privacy_note": "The imported method fixes clipping/noise but supplies no epsilon accountant.",
            })
        write_json(directory / "steering_config.json", metadata)
        save_audit_splits(directory, selected, validation, test)

    report = {
        **common,
        "arguments": vars(args),
        "elapsed_seconds": time.time() - started,
        "mean_dir": str((root / "mean").resolve()) if "mean" in args.variants else None,
        "private_dir": str((root / "private").resolve()) if "private" in args.variants else None,
    }
    write_json(root / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_chatglm_base(args.base_model)
    train = load_split(args.data_dir, "train")
    for split in ("validation", "test"):
        load_split(args.data_dir, split)
    validate_args(args, parser, len(train))
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
