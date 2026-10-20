#!/usr/bin/env python3
"""Train a generative ChatGLM3 LoRA with Opacus DP-SGD."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import time
from types import MethodType

from .common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_RAW_DATA,
    load_split,
    prepare_training_splits,
    require_empty_output,
    save_audit_splits,
    validate_chatglm_base,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", default="outputs/models/gradient")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8, help="Logical DP batch")
    parser.add_argument("--max-physical-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    positive = ("rank", "lora_alpha", "epochs", "batch_size", "max_physical_batch_size",
                "max_length", "max_train_samples", "eval_samples")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("counts and dimensions must be positive")
    if args.max_physical_batch_size > args.batch_size:
        parser.error("physical batch cannot exceed logical batch")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.epsilon <= 0:
        parser.error("optimizer and privacy parameters must be positive")
    if not 0 < args.delta < 1 or not 0 <= args.lora_dropout < 1:
        parser.error("delta or dropout is invalid")


def chatglm_lora_forward(self, x, *args, **kwargs):
    result = self._sequence_first_forward(x.transpose(0, 1), *args, **kwargs)
    return result.transpose(0, 1)


def run(args):
    import numpy as np
    import torch
    try:
        import opacus
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
    except ImportError as error:
        raise RuntimeError("Gradient protection requires opacus==1.5.4") from error
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.tuners.lora import LoraLayer
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from .opacus_compat import apply_bfloat16_patches

    if not torch.cuda.is_available():
        raise RuntimeError("DP-SGD training requires CUDA")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id {args.gpu_id} is unavailable")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bfloat16")
    dtype = getattr(torch, args.dtype)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    apply_bfloat16_patches()

    output_dir = require_empty_output(args.output_dir)
    train_rows, validation_rows, test_rows = prepare_training_splits(
        args.data_dir, args.max_train_samples, args.seed,
    )
    audit_dir, _ = save_audit_splits(output_dir, train_rows, validation_rows, test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    config.seq_length = args.max_length
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ))
    for module in model.modules():
        if isinstance(module, LoraLayer) and not hasattr(module, "_sequence_first_forward"):
            module._sequence_first_forward = module.forward
            module.forward = MethodType(chatglm_lora_forward, module)
    model.to(device)
    model.print_trainable_parameters()

    class TokenDataset(Dataset):
        def __init__(self, rows):
            encoded = tokenizer([row["text"] for row in rows], padding="max_length",
                                truncation=True, max_length=args.max_length, return_tensors="pt")
            self.input_ids = encoded["input_ids"]
            self.attention_mask = encoded["attention_mask"]
            self.labels = self.input_ids.clone()
            self.labels[self.attention_mask == 0] = -100

        def __len__(self):
            return self.input_ids.shape[0]

        def __getitem__(self, index):
            return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index],
                    "labels": self.labels[index]}

    train_dataset = TokenDataset(train_rows)
    train_eval = TokenDataset(train_rows[:args.eval_samples])
    validation = TokenDataset(validation_rows[:args.eval_samples])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              generator=torch.Generator().manual_seed(args.seed + 100_000))
    train_eval_loader = DataLoader(train_eval, batch_size=1)
    validation_loader = DataLoader(validation, batch_size=1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        target_epsilon=args.epsilon, target_delta=args.delta, epochs=args.epochs,
        max_grad_norm=args.max_grad_norm, grad_sample_mode="hooks", poisson_sampling=True,
    )

    @torch.inference_mode()
    def evaluate(loader):
        model.eval()
        losses = []
        for batch in loader:
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss.float()
            losses.append(float(loss.cpu()))
        model.train()
        value = sum(losses) / len(losses)
        return {"loss": value, "perplexity": math.exp(min(value, 20))}

    history = []
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        steps = 0
        with BatchMemoryManager(data_loader=train_loader,
                                max_physical_batch_size=args.max_physical_batch_size,
                                optimizer=optimizer) as memory_loader:
            for batch in tqdm(memory_loader, desc=f"DP epoch {epoch + 1}/{args.epochs}"):
                if batch["input_ids"].shape[0] == 0:
                    continue
                loss = model(**{key: value.to(device) for key, value in batch.items()}).loss.float()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite DP loss at epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach().cpu())
                steps += 1
        train_metrics = evaluate(train_eval_loader)
        validation_metrics = evaluate(validation_loader)
        record = {"epoch": epoch + 1, "online_train_loss": total / max(steps, 1),
                  "epsilon": float(privacy_engine.get_epsilon(args.delta)), "delta": args.delta,
                  "train": train_metrics, "validation": validation_metrics,
                  "generalization_gap": validation_metrics["loss"] - train_metrics["loss"]}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    raw_model = getattr(model, "_module", model)
    raw_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    dtypes = Counter(str(p.dtype) for p in raw_model.parameters() if p.requires_grad)
    report = {
        "purpose": "gradient_protected_causal_lm",
        "adapter_task": "CAUSAL_LM",
        "arguments": vars(args),
        "train_records": len(train_rows),
        "audit_data_dir": str(audit_dir.resolve()),
        "trainable_parameter_tensor_dtypes": dict(dtypes),
        "privacy": {
            "accountant": type(privacy_engine.accountant).__name__,
            "target_epsilon": args.epsilon,
            "actual_epsilon": float(privacy_engine.get_epsilon(args.delta)),
            "delta": args.delta,
            "noise_multiplier": float(optimizer.noise_multiplier),
            "max_grad_norm": args.max_grad_norm,
        },
        "opacus_version": opacus.__version__,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    write_json(output_dir / "training_report.json", report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    validate_chatglm_base(args.base_model)
    for split in ("train", "validation", "test"):
        load_split(args.data_dir, split)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
