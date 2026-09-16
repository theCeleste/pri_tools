#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import logging
from transformers import TrainerCallback
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from sklearn.metrics import accuracy_score, f1_score
import hashlib


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sst2_from_tsv(path: str, valid_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    从本地 train.tsv 加载 SST-2，并切分出验证集。
    确保 label 列是 {0,1} 的整型。
    """
    print(f"[Data] Loading SST-2 from: {path}")
    df = pd.read_csv(path, sep="\t", header=0)

 
    if "sentence" not in df.columns or "label" not in df.columns:
        if df.shape[1] >= 2:
            df.columns = ["sentence", "label"] + list(df.columns[2:])
        else:
            raise ValueError("train.tsv 至少需要两列：sentence 和 label")

    print(f"[Data] 原始行数: {len(df)}")
    print(f"[Data] label 列前几行:\n{df['label'].head()}")

 
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

 
    before_drop_na = len(df)
    df = df.dropna(subset=["label"])
    print(f"[Data] 去掉 label=NaN 的行数: {before_drop_na - len(df)}")

 
    df["label"] = df["label"].astype(int)

 
    before_filter = len(df)
    valid_labels = [0, 1]
    mask_valid = df["label"].isin(valid_labels)
    if not mask_valid.all():
        print("[Data] 发现非法 label（不在 0/1 中），将会被丢弃:")
        print(df.loc[~mask_valid, "label"].value_counts())
    df = df[mask_valid].reset_index(drop=True)
    print(f"[Data] 丢弃非法 label 行数: {before_filter - len(df)}")

    print("[Data] 清洗后 label 分布:")
    print(df["label"].value_counts())

 
    print("[Data] 清洗后训练集前 3 行:")
    print(df[["sentence", "label"]].head(3))

 
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

    return DatasetDict(
        {
            "train": train_dataset,
            "validation": valid_dataset,
        }
    )


def prepare_tokenizer(model_name_or_path: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_datasets(datasets: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def preprocess(examples):
        result = tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
        )
 
        result["labels"] = examples["label"]
        return result

    tokenized = datasets.map(
        preprocess,
        batched=True,
        remove_columns=[col for col in datasets["train"].column_names if col not in ["sentence", "label"]],
    )
 
    tokenized = tokenized.remove_columns("sentence")
    return tokenized


def build_model_with_lora(
    model_name_or_path: str,
    num_labels: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):

 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
 
    if hasattr(config, "gradient_checkpointing"):
        config.gradient_checkpointing = False

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16, 
 
 
    )

 
    if model.config.pad_token_id is None and hasattr(model.config, "eos_token_id"):
        model.config.pad_token_id = model.config.eos_token_id

 
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

 
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "gradient_checkpointing_enable"):
 
        model.gradient_checkpointing_disable()

 
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

 
    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = False

    return model


def build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        return {
            "accuracy": acc,
            "f1": f1,
        }

    return compute_metrics


def hash_lora_parameters(model: PeftModel) -> str:

    m = hashlib.sha256()
    total = 0
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            arr = p.detach().cpu().numpy()
            m.update(arr.tobytes())
            total += arr.size
    if total == 0:
        return "NO_LORA_PARAMS"
    return m.hexdigest()
class EpochBatchStatsCallback(TrainerCallback):
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

        self.epoch_batch_stats = []
        self._t0 = None
        self._epoch_index = 0
        self._mem_before_bytes = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.epoch_jsonl_path = os.path.join(self.output_dir, "epoch_stats.jsonl")

    @staticmethod
    def _gb(x: int) -> float:
        return float(x) / (1024 ** 3)

    def _begin_measure(self):
        self._t0 = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self._mem_before_bytes = torch.cuda.memory_allocated()
            except Exception:
                self._mem_before_bytes = None

    def _end_measure_and_record(self, args, state, inputs=None):
        if self._t0 is None:
            return

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        batch_time = time.perf_counter() - self._t0
        self._t0 = None

        # batch_size：优先从 inputs 推断；否则用估算
        batch_size = None
        if isinstance(inputs, dict):
            if "input_ids" in inputs and hasattr(inputs["input_ids"], "size"):
                batch_size = int(inputs["input_ids"].size(0))
            elif "labels" in inputs and hasattr(inputs["labels"], "size"):
                batch_size = int(inputs["labels"].size(0))

        world_size = int(getattr(state, "world_size", 1) or 1)

        if batch_size is None:
            # 如果是 on_step_end（optimizer step），会包含梯度累积的多个 micro-batch
            # 这里按 step 的处理量估算
            ga = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
            batch_size = int(args.per_device_train_batch_size) * world_size * ga

        memory_before_gb = self._gb(self._mem_before_bytes) if self._mem_before_bytes is not None else 0.0
        memory_after_gb = 0.0
        memory_peak_gb = 0.0
        if torch.cuda.is_available():
            try:
                memory_after_gb = self._gb(torch.cuda.memory_allocated())
                memory_peak_gb = self._gb(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        throughput = (batch_size / batch_time) if batch_time > 0 else 0.0

        self.epoch_batch_stats.append({
            "batch_time": float(batch_time),
            "throughput": float(throughput),
            "batch_size": int(batch_size),
            "memory_before_gb": float(memory_before_gb),
            "memory_after_gb": float(memory_after_gb),
            "memory_peak_gb": float(memory_peak_gb),
        })

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_index += 1
        self.epoch_batch_stats = []
        self.logger.info(f"Epoch {self._epoch_index} begin.")

    def on_step_begin(self, args, state, control, **kwargs):
        self._begin_measure()

    def on_step_end(self, args, state, control, **kwargs):
        self._end_measure_and_record(args, state, inputs=kwargs.get("inputs"))

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = self._epoch_index
        stats = self.epoch_batch_stats

        if stats:
            epoch_stats = {
                "epoch": epoch,
                "total_batches": len(stats),
                "avg_batch_time": sum(s["batch_time"] for s in stats) / len(stats),
                "avg_throughput": sum(s["throughput"] for s in stats) / len(stats),
                "total_samples": sum(s["batch_size"] for s in stats),
                "total_time": sum(s["batch_time"] for s in stats),
                "avg_memory_gb": sum(s["memory_after_gb"] for s in stats) / len(stats),
                "max_memory_gb": max(s["memory_peak_gb"] for s in stats),
                "min_memory_gb": min(s["memory_before_gb"] for s in stats),
            }

            # self.logger.info(
            #     f"Epoch {epoch} Summary: "
            #     f"Avg Time={epoch_stats['avg_batch_time']:.4f}s/batch, "
            #     f"Avg Throughput={epoch_stats['avg_throughput']:.2f} samples/s, "
            #     f"Total Samples={epoch_stats['total_samples']}, "
            #     f"Total Time={epoch_stats['total_time']:.2f}s, "
            #     f"Avg Memory={epoch_stats['avg_memory_gb']:.2f}GB, "
            #     f"Peak Memory={epoch_stats['max_memory_gb']:.2f}GB, "
            #     f"Min Memory={epoch_stats['min_memory_gb']:.2f}GB"
            # )

            try:
                with open(self.epoch_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(epoch_stats, ensure_ascii=False) + "\n")
            except Exception as e:
                self.logger.warning(f"Failed to write epoch stats jsonl: {e}")

        else:
            self.logger.warning(
                f"Epoch {epoch} has no batch stats collected. "
                f"This usually means batch hooks were not triggered; step hooks should have collected them—"
                f"if not, check transformers version / callback registration."
            )

        self.logger.info(f"Epoch {epoch} end.")

def main():
    parser = argparse.ArgumentParser()

 
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/path/to/llama-2-7b-hf",
        help="本地 LLaMA2-7B HF 格式权重路径",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="/home/wpj/ptcds/PTCDS/data/SST-2/train.tsv",
        help="SST-2 train.tsv 文件路径（baseline: 原始；esp3: PTCDS 后）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/llama2-7b-sst2-lora-base",
        help="输出目录（checkpoint、日志等）",
    )

 
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

 
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

 
    parser.add_argument("--wandb_project", type=str, default="llama2-sst2-lora")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_off", action="store_true")

    parser.add_argument("--run_name", type=str, default="llama2-7b-sst2-lora")

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(f"[Config] model_name_or_path = {args.model_name_or_path}")
    print(f"[Config] train_path        = {args.train_path}")
    print(f"[Config] output_dir        = {args.output_dir}")
    print(f"[Config] run_name          = {args.run_name}")

 
    os.environ["WANDB_DISABLED"] = "true"
    report_to = []

    # if args.wandb_off:
    #     os.environ["WANDB_DISABLED"] = "true"
    #     report_to = []
    # else:
    #     os.environ["WANDB_PROJECT"] = args.wandb_project
    #     if args.wandb_entity is not None:
    #         os.environ["WANDB_ENTITY"] = args.wandb_entity
    #     report_to = ["wandb"]

    set_seed(args.seed)

 
    raw_datasets = load_sst2_from_tsv(
        args.train_path,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

 
    tokenizer = prepare_tokenizer(args.model_name_or_path, args.max_length)
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer)

 
    print("Building model with LoRA and flash_attention_2...")
    model = build_model_with_lora(
        model_name_or_path=args.model_name_or_path,
        num_labels=2,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

 
    if isinstance(model, PeftModel):
        h_before = hash_lora_parameters(model)
        print(f"[LoRA] 训练前 LoRA SHA256 = {h_before}")
    else:
        print("[LoRA] 警告: model 不是 PeftModel，LoRA 没挂上。")

 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,

 
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        disable_tqdm=False,

        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, "logs"),

        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        optim="adamw_torch",

        report_to=report_to,
        gradient_checkpointing=False,
    )
    compute_metrics = build_compute_metrics()

 
    stats_cb = EpochBatchStatsCallback(output_dir=args.output_dir, logger=logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[stats_cb],
    )

    print(">>> Trainer created, start training...")
    train_result = trainer.train()
    print(">>> Training finished.")

    # trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

 
    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir) 

    print(f"Best model saved to: {best_model_dir}")

    if isinstance(trainer.model, PeftModel):
        h_after = hash_lora_parameters(trainer.model)
        print(f"[LoRA] 训练后 LoRA SHA256 = {h_after}")
        with open(os.path.join(best_model_dir, "lora_hash.txt"), "w") as f:
            f.write(h_after + "\n")
    else:
        print("LoRA 没有正确套上。")


if __name__ == "__main__":
    main()
"""
运行命令：


CUDA_VISIBLE_DEVICES=1 python /home/wpj/ptcds/PTCDS/finetune_llama2_sst2_lora_data.py --model_name_or_path /home/wpj/model/llama2-7b --train_path ./output_PTCDS/SST-2_glove_eps6/eps_6.00/sword_0.50_p_0.20/train.tsv --output_dir /home/wpj/ptcds/outputs/llama2-sst2-lora-esp4-3.4 --eval_steps 948 --wandb_project llama2-sst2-lora --run_name llama2-7b-sst2
#eps n
CUDA_VISIBLE_DEVICES=0 python /home/wpj/ptcds/PTCDS/finetune_llama2_sst2_lora_data.py --model_name_or_path /home/wpj/model/llama2-7b --train_path ./output_PTCDS_plus_glove/SST-2/eps_6.00/sword_0.90_p_0.30/train.tsv --output_dir /home/wpj/ptcds/outputs/llama2-sst2-lora-esp6-4.20 --eval_steps 948 --wandb_project llama2-sst2-lora --run_name llama2-7b-sst2
#base
CUDA_VISIBLE_DEVICES=3 python /home/wpj/ptcds/PTCDS/finetune_llama2_sst2_lora_data.py --model_name_or_path /home/wpj/model/llama2-7b --train_path /home/wpj/ptcds/PTCDS/data/SST-2/train.tsv --output_dir /home/wpj/ptcds/outputs/llama2-sst2-lora-base-4.20 --eval_steps 948 --wandb_project llama2-sst2-lora --run_name llama2-7b-sst2

"""