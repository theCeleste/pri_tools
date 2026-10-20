#!/usr/bin/env python3
"""Deliberately overfit a ChatGLM3 CAUSAL_LM LoRA for extraction auditing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--original-data-dir")
    parser.add_argument("--output-dir", default="outputs/models/raw")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--max-train-samples", type=int, default=0,
        help="Maximum training records; 0 uses the complete training split",
    )
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = ("rank", "lora_alpha", "epochs", "batch_size", "gradient_accumulation",
                "max_length", "eval_samples")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("training counts and dimensions must be positive")
    if args.max_train_samples < 0:
        parser.error("max-train-samples must be nonnegative (0 means the full training split)")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("invalid optimizer or LoRA parameters")


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("ChatGLM3 training requires CUDA")
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
    train_rows, validation_rows, test_rows = prepare_training_splits(
        args.data_dir, args.max_train_samples, args.seed,
    )
    print(json.dumps({
        "training_data": str(Path(args.data_dir).resolve()),
        "train_records": len(train_rows),
        "selection": "full_training_split" if args.max_train_samples == 0 else "seeded_subset",
    }, ensure_ascii=False), flush=True)
    audit_dir, original_dir = save_audit_splits(
        output_dir, train_rows, validation_rows, test_rows, args.original_data_dir,
    )

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
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )).to(device)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.print_trainable_parameters()

    class TextDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]["text"]

    def collate(texts):
        encoded = tokenizer(list(texts), return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_length)
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    train_eval_loader = DataLoader(TextDataset(train_rows[:args.eval_samples]), batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate)
    validation_loader = DataLoader(TextDataset(validation_rows[:args.eval_samples]), batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    @torch.inference_mode()
    def evaluate(loader):
        model.eval()
        losses = []
        for batch in loader:
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss.float()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss")
            losses.append(float(loss.cpu()))
        model.train()
        value = sum(losses) / len(losses)
        return {"loss": value, "perplexity": math.exp(min(value, 20))}

    history = []
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"), 1):
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss.float()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
            (loss / args.gradient_accumulation).backward()
            total += float(loss.detach().cpu())
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        train_metrics = evaluate(train_eval_loader)
        validation_metrics = evaluate(validation_loader)
        record = {
            "epoch": epoch + 1,
            "online_train_loss": total / len(train_loader),
            "train": train_metrics,
            "validation": validation_metrics,
            "generalization_gap": validation_metrics["loss"] - train_metrics["loss"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    report = {
        "purpose": "overfit_causal_lm_for_authorized_extraction_audit",
        "adapter_task": "CAUSAL_LM",
        "arguments": vars(args),
        "train_records": len(train_rows),
        "audit_data_dir": str(audit_dir.resolve()),
        "audit_original_data_dir": str(original_dir.resolve()) if original_dir else None,
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
        if args.original_data_dir:
            load_split(args.original_data_dir, split)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return vars(args)
    return run(args)


if __name__ == "__main__":
    main()
