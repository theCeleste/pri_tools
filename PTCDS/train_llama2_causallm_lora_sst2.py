#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import inspect
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ------------------------
# Utils
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for u in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.2f} {u}"
    return f"{n:.2f} PB"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_model_report(model: torch.nn.Module, tokenizer, args):
    print("\n================= Model / Tokenizer Report =================")
    print(f"[Device] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device] cuda_name={torch.cuda.get_device_name(0)}")
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"[Device] cuda_mem_free={human_bytes(int(free))}, cuda_mem_total={human_bytes(int(total))}")
        except Exception:
            pass

    print(f"[Tokenizer] name_or_path={getattr(tokenizer, 'name_or_path', 'N/A')}")
    print(f"[Tokenizer] vocab_size={tokenizer.vocab_size}")
    print(f"[Tokenizer] pad_token={tokenizer.pad_token} pad_token_id={tokenizer.pad_token_id}")
    print(f"[Tokenizer] eos_token={tokenizer.eos_token} eos_token_id={tokenizer.eos_token_id}")
    print(f"[Tokenizer] bos_token={tokenizer.bos_token} bos_token_id={tokenizer.bos_token_id}")
    print(f"[Tokenizer] model_max_length={tokenizer.model_max_length} padding_side={tokenizer.padding_side}")

    # Config summary
    cfg = model.config
    print(f"[Config] model_type={getattr(cfg, 'model_type', 'N/A')}")
    print(f"[Config] hidden_size={getattr(cfg, 'hidden_size', 'N/A')}, num_hidden_layers={getattr(cfg, 'num_hidden_layers', 'N/A')}")
    print(f"[Config] num_attention_heads={getattr(cfg, 'num_attention_heads', 'N/A')}")
    print(f"[Config] vocab_size={getattr(cfg, 'vocab_size', 'N/A')}")
    print(f"[Config] use_cache={getattr(cfg, 'use_cache', 'N/A')}")
    print(f"[Config] torch_dtype={getattr(cfg, 'torch_dtype', 'N/A')}")
    print(f"[Train] gradient_checkpointing={args.gradient_checkpointing}")
    print(f"[Train] bf16={args.bf16} fp16={args.fp16}")

    # Parameter counts
    cnt = count_parameters(model)
    print(f"[Params] total={cnt['total']:,}  trainable={cnt['trainable']:,}  trainable_ratio={cnt['trainable']/cnt['total']*100:.4f}%")

    # If PEFT
    if isinstance(model, PeftModel):
        print("[PEFT] Detected PeftModel (LoRA attached)")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        # Show some trainable params
        print("[PEFT] Trainable parameter tensors (first 25):")
        shown = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f"  - {name}: shape={tuple(p.shape)} dtype={p.dtype}")
                shown += 1
                if shown >= 25:
                    break

    print("============================================================\n")


# ------------------------
# Data
# ------------------------
def load_sst2_tsv_for_lm(path: str, text_col: str, valid_ratio: float, seed: int, max_samples: Optional[int] = None) -> DatasetDict:
    """
    读取 SST-2 tsv（或任意含文本列的 tsv），仅使用 text_col 作为 LM 训练文本。
    """
    print(f"[Data] Loading TSV: {path}")
    df = pd.read_csv(path, sep="\t", header=0)
    if text_col not in df.columns:
        raise ValueError(f"[Data] text_col='{text_col}' not found. Columns={list(df.columns)}")

    df = df[[text_col]].copy()
    df[text_col] = df[text_col].astype(str)

    # 简单过滤空行
    before = len(df)
    df = df[df[text_col].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"[Data] rows: {before} -> {len(df)} (drop empty)")

    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)
        print(f"[Data] using first max_samples={max_samples}, rows={len(df)}")

    train_df, valid_df = train_test_split(df, test_size=valid_ratio, random_state=seed)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
        }
    )


def prepare_tokenizer(local_tokenizer_path: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_fast=True, local_files_only=True)

    # LLaMA 通常没 pad_token，设为 eos_token；注意：collator 会把 pad positions 的 labels 置为 -100 避免学 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"  # 训练 teacher-forcing 用右侧 padding 没问题
    return tokenizer


def tokenize_for_causallm(
    datasets: DatasetDict,
    tokenizer,
    text_col: str,
    max_length: int,
    add_eos: bool = True,
) -> DatasetDict:
    eos = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    def preprocess(examples):
        texts = examples[text_col]
        if add_eos and eos:
            texts = [t + eos for t in texts]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        return enc

    tokenized = datasets.map(preprocess, batched=True, remove_columns=[text_col])
    return tokenized


# ------------------------
# Data collator (important!)
# ------------------------
@dataclass
class DataCollatorForCausalLMWithPadding:
    """
    - 动态 padding
    - labels = input_ids
    - padding 部分 labels = -100（保证不会在 pad 上计算 loss）
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore pad

        batch["labels"] = labels
        return batch


# ------------------------
# Model + LoRA
# ------------------------
def build_causallm_with_lora(
    model_path: str,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    # 训练建议关闭 cache
    if hasattr(config, "use_cache"):
        config.use_cache = False

    # 尽量避免 flash attention：不传 attn_implementation="flash_attention_2"
    # 有些版本支持 attn_implementation="eager"，我们做兼容传参
    from_pretrained_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    kwargs = dict(
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if "attn_implementation" in from_pretrained_sig.parameters:
        # 强制 eager（不依赖 flash-attn）
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    # pad_token_id
    if model.config.pad_token_id is None and hasattr(model.config, "eos_token_id"):
        model.config.pad_token_id = model.config.eos_token_id

    # gradient checkpointing
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    return model


def build_training_args(args) -> TrainingArguments:
    # transformers 版本兼容：evaluation_strategy vs eval_strategy
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs = dict(
        output_dir=args.output_dir,
        run_name=args.run_name,

        # eval / log / save
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # train hyperparams
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # precision
        bf16=args.bf16,
        fp16=args.fp16,

        # perf
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,

        report_to=args.report_to,
        disable_tqdm=False,

        gradient_checkpointing=args.gradient_checkpointing,
    )

    if "eval_strategy" in ta_sig.parameters and "evaluation_strategy" not in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")

    # 一些版本叫 logging_dir 存在；没有也无所谓
    if "logging_dir" in ta_sig.parameters:
        ta_kwargs["logging_dir"] = os.path.join(args.output_dir, "logs")

    return TrainingArguments(**ta_kwargs)


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--model_path", type=str, required=True, help="本地 LLaMA2 HF 目录，如 /home/wpj/model/llama2-7b")
    ap.add_argument("--tokenizer_path", type=str, default=None, help="本地 tokenizer 目录；默认等于 model_path")
    ap.add_argument("--train_tsv", type=str, required=True, help="SST-2 train.tsv（或任意含 sentence 列的 tsv）")
    ap.add_argument("--text_col", type=str, default="sentence", help="文本列名，SST-2 通常是 sentence")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="llama2-causallm-lora")

    # data
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--max_train_samples", type=int, default=0, help=">0 则只用前 N 条做快速验证")
    ap.add_argument("--add_eos", action="store_true", help="在每条文本后追加 eos_token（推荐打开）")

    # train
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--dataloader_num_workers", type=int, default=2)
    ap.add_argument("--optim", type=str, default="adamw_torch")

    # precision / memory
    ap.add_argument("--bf16", action="store_true", help="A100 推荐开 bf16")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存，建议单卡训练开启")

    # lora
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="逗号分隔；LLaMA2 常用这些",
    )

    # wandb
    ap.add_argument("--wandb_project", type=str, default="llama2-causallm-lora")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_off", action="store_true")

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    set_seed(args.seed)

    # W&B
    if args.wandb_off:
        os.environ["WANDB_DISABLED"] = "true"
        args.report_to = []
    else:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        args.report_to = ["wandb"]

    # dtype
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("========== Run Config ==========")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("================================\n")

    # data
    raw_ds = load_sst2_tsv_for_lm(
        path=args.train_tsv,
        text_col=args.text_col,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        max_samples=(args.max_train_samples if args.max_train_samples > 0 else None),
    )

    # tokenizer
    tokenizer = prepare_tokenizer(args.tokenizer_path, args.max_length)

    # tokenize
    tokenized = tokenize_for_causallm(
        datasets=raw_ds,
        tokenizer=tokenizer,
        text_col=args.text_col,
        max_length=args.max_length,
        add_eos=args.add_eos,
    )

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # model
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model = build_causallm_with_lora(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    # Print detailed model info
    print_model_report(model, tokenizer, args)

    # training args
    training_args = build_training_args(args)

    # Trainer (tqdm is enabled by default)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print(">>> Trainer created. Start training ...")
    train_result = trainer.train()
    print(">>> Training finished.")

    # Save
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Save best / final adapter
    final_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)  # for PEFT: saves adapter weights
    print(f"[Save] final adapter saved to: {final_dir}")

    # Save tokenizer too (optional but recommended)
    tok_dir = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)
    print(f"[Save] tokenizer saved to: {tok_dir}")


if __name__ == "__main__":
    main()
