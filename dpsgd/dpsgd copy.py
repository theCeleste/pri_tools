"""
DP-SGD Fine-tuning of LLaMA2-7B on SST-2
显卡: A100 80GB
算法: DP-SGD (Differentially Private SGD) via Opacus
数据集: SST-2 (Stanford Sentiment Treebank)

依赖安装:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets opacus peft accelerate bitsandbytes
    pip install scikit-learn tqdm
"""

import os
# 必须在 tokenizers 库被导入之前设置，避免 DataLoader fork 后出现死锁警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import math
import time
import json
import threading
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────── 日志配置 ───────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────── 性能监控器 ───────────────────────────
class PerformanceProfiler:
    """
    实时追踪训练过程中的：
      - GPU 显存占用（当前 / 峰值，MB）
      - CPU 内存占用（RSS，MB）
      - 运行时间（总时长 / 各阶段耗时，秒）
      - 吞吐率（samples/sec，逻辑 batch 维度）
    后台线程每 interval 秒采样一次，不影响主训练流程。
    """

    def __init__(self, interval: float = 1.0, output_dir: str = "."):
        self.interval   = interval
        self.output_dir = output_dir
        self._lock      = threading.Lock()

        # ── 显存 ──
        self.gpu_available = torch.cuda.is_available()
        self.peak_gpu_mb   = 0.0        # 全局峰值显存（MB）
        self.cur_gpu_mb    = 0.0        # 最近一次采样显存（MB）

        # ── CPU 内存 ──
        self.peak_cpu_mb   = 0.0        # 全局峰值 RSS（MB）
        self.cur_cpu_mb    = 0.0

        # ── 时间 ──
        self.wall_start = None          # train() 开始时刻
        self.stage_times: Dict[str, float] = {}  # {阶段名: 耗时秒}

        # ── 吞吐 ──
        self.total_samples   = 0        # 已处理逻辑样本数
        self.throughput_log: List[Dict] = []     # 逐步记录，最终写 CSV

        # ── 后台采样线程 ──
        self._running = False
        self._thread  = None
        self._sample_log: List[Dict] = []

    # ──────── 后台采样 ────────
    def _sample_loop(self):
        process = psutil.Process()
        while self._running:
            ts       = time.perf_counter()
            cpu_mb   = process.memory_info().rss / 1024 ** 2
            gpu_mb   = 0.0
            if self.gpu_available:
                gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2

            with self._lock:
                self.cur_cpu_mb  = cpu_mb
                self.cur_gpu_mb  = gpu_mb
                self.peak_cpu_mb = max(self.peak_cpu_mb, cpu_mb)
                self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)
                self._sample_log.append({
                    "timestamp_s":  round(ts - (self.wall_start or ts), 3),
                    "cpu_rss_mb":   round(cpu_mb, 1),
                    "gpu_alloc_mb": round(gpu_mb, 1),
                })

            time.sleep(self.interval)

    def start(self):
        """训练开始时调用"""
        self.wall_start = time.perf_counter()
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
        self._running = True
        self._thread  = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("PerformanceProfiler 已启动（后台采样间隔 %.1fs）", self.interval)

    def stop(self):
        """训练结束时调用"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        # 从 PyTorch 拿精确峰值（比轮询更准）
        if self.gpu_available:
            torch_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            with self._lock:
                self.peak_gpu_mb = max(self.peak_gpu_mb, torch_peak)

    # ──────── 阶段计时 ────────
    def begin_stage(self, name: str) -> float:
        """返回开始时间戳，供 end_stage 使用"""
        return time.perf_counter()

    def end_stage(self, name: str, t0: float):
        elapsed = time.perf_counter() - t0
        with self._lock:
            self.stage_times[name] = self.stage_times.get(name, 0.0) + elapsed

    # ──────── 吞吐记录 ────────
    def log_step(self, global_step: int, samples_in_step: int, step_time_s: float):
        """每个逻辑 step 结束后调用"""
        throughput = samples_in_step / step_time_s if step_time_s > 0 else 0.0
        with self._lock:
            self.total_samples += samples_in_step
            self.throughput_log.append({
                "global_step":     global_step,
                "samples":         samples_in_step,
                "step_time_s":     round(step_time_s, 4),
                "throughput_sps":  round(throughput, 2),
                "gpu_alloc_mb":    round(self.cur_gpu_mb, 1),
                "cpu_rss_mb":      round(self.cur_cpu_mb, 1),
            })

    # ──────── 汇总 ────────
    def total_wall_time(self) -> float:
        return (time.perf_counter() - self.wall_start) if self.wall_start else 0.0

    def avg_throughput(self) -> float:
        if not self.throughput_log:
            return 0.0
        return float(np.mean([r["throughput_sps"] for r in self.throughput_log]))

    def peak_throughput(self) -> float:
        if not self.throughput_log:
            return 0.0
        return float(np.max([r["throughput_sps"] for r in self.throughput_log]))

    def summary(self) -> Dict:
        wall = self.total_wall_time()
        overall_sps = self.total_samples / wall if wall > 0 else 0.0
        return {
            "wall_time_s":           round(wall, 2),
            "wall_time_hms":         _fmt_hms(wall),
            "total_samples":         self.total_samples,
            "overall_throughput":    round(overall_sps, 2),
            "avg_step_throughput":   round(self.avg_throughput(), 2),
            "peak_step_throughput":  round(self.peak_throughput(), 2),
            "peak_gpu_mb":           round(self.peak_gpu_mb, 1),
            "peak_cpu_rss_mb":       round(self.peak_cpu_mb, 1),
            "stage_times":           {k: round(v, 2) for k, v in self.stage_times.items()},
        }

    def print_summary(self):
        s = self.summary()
        sep = "=" * 64
        logger.info(sep)
        logger.info("  *** 性能统计报告 ***")
        logger.info(sep)
        logger.info(f"  总运行时间           : {s['wall_time_hms']}  ({s['wall_time_s']:.1f} s)")
        logger.info(f"  处理样本总数         : {s['total_samples']:,}")
        logger.info(f"  整体吞吐率           : {s['overall_throughput']:.2f} samples/sec")
        logger.info(f"  平均逐步吞吐率       : {s['avg_step_throughput']:.2f} samples/sec")
        logger.info(f"  峰值逐步吞吐率       : {s['peak_step_throughput']:.2f} samples/sec")
        logger.info(f"  GPU 显存峰值         : {s['peak_gpu_mb']:.1f} MB  "
                    f"({s['peak_gpu_mb']/1024:.2f} GB)")
        logger.info(f"  CPU 内存峰值 (RSS)   : {s['peak_cpu_rss_mb']:.1f} MB  "
                    f"({s['peak_cpu_rss_mb']/1024:.2f} GB)")
        if s["stage_times"]:
            logger.info("  各阶段耗时:")
            for stage, t in s["stage_times"].items():
                logger.info(f"    {stage:<28}: {_fmt_hms(t)}  ({t:.1f} s)")
        logger.info(sep)

    def save_reports(self):
        """保存 JSON 摘要 + 逐步吞吐 CSV + 内存时序 CSV"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 1) JSON 摘要
        json_path = os.path.join(self.output_dir, "perf_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)

        # 2) 逐步吞吐 CSV
        if self.throughput_log:
            csv_path = os.path.join(self.output_dir, "perf_throughput.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.throughput_log[0].keys())
                writer.writeheader()
                writer.writerows(self.throughput_log)

        # 3) 内存时序 CSV
        if self._sample_log:
            mem_path = os.path.join(self.output_dir, "perf_memory_timeline.csv")
            with open(mem_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._sample_log[0].keys())
                writer.writeheader()
                writer.writerows(self._sample_log)

        logger.info(
            "性能报告已保存:\n"
            f"  {json_path}\n"
            f"  {self.output_dir}/perf_throughput.csv\n"
            f"  {self.output_dir}/perf_memory_timeline.csv"
        )


def _fmt_hms(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────── 超参数 ───────────────────────────
@dataclass
class TrainingConfig:
    # 模型
    model_name: str = "./llama2-7b"   # 本地模型目录，也可通过 --model_name 指定绝对路径
    num_labels: int = 2                      # SST-2 二分类

    # 数据
    max_seq_length: int = 128                # SST-2 句子较短，128 足够
    dataset_name: str = "sst2"
    dataset_path: str = ""

    # DP 隐私预算
    target_epsilon: float = 8.0              # ε，越小隐私越强
    target_delta: float = 1e-5              # δ，通常设为 1/N
    max_grad_norm: float = 1.0              # 梯度裁剪范数 C

    # 训练
    num_epochs: int = 1
    per_device_batch_size: int = 8          # 物理 batch size（受显存限制）
    max_physical_batch_size: int = 8        # Opacus BatchMemoryManager 用
    logical_batch_size: int = 256           # 逻辑 batch size（影响噪声量）
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 42

    # LoRA（大幅减少可训练参数，降低隐私成本）
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # 输出
    output_dir: str = "./dpsgd_llama2_sst2_output"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 20

    # 硬件
    fp16: bool = True                        # A100 支持 BF16/FP16
    bf16: bool = False                       # 与 fp16 二选一
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = False     # Opacus 与 grad checkpointing 有限兼容

    # 快速验证（smoke test）
    smoke_test: bool = False   # --smoke_test：只取少量样本，验证流程是否跑通


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="DP-SGD fine-tuning LLaMA2-7B on SST-2")
    cfg = TrainingConfig()

    parser.add_argument("--model_name",            default=cfg.model_name)
    parser.add_argument("--dataset_path",           default=cfg.dataset_path)
    parser.add_argument("--max_seq_length",        type=int,   default=cfg.max_seq_length)
    parser.add_argument("--target_epsilon",        type=float, default=cfg.target_epsilon)
    parser.add_argument("--target_delta",          type=float, default=cfg.target_delta)
    parser.add_argument("--max_grad_norm",         type=float, default=cfg.max_grad_norm)
    parser.add_argument("--num_epochs",            type=int,   default=cfg.num_epochs)
    parser.add_argument("--per_device_batch_size", type=int,   default=cfg.per_device_batch_size)
    parser.add_argument("--max_physical_batch_size", type=int, default=cfg.max_physical_batch_size)
    parser.add_argument("--logical_batch_size",    type=int,   default=cfg.logical_batch_size)
    parser.add_argument("--learning_rate",         type=float, default=cfg.learning_rate)
    parser.add_argument("--seed",                  type=int,   default=cfg.seed)
    parser.add_argument("--use_lora",              action="store_true", default=cfg.use_lora)
    parser.add_argument("--lora_r",                type=int,   default=cfg.lora_r)
    parser.add_argument("--output_dir",            default=cfg.output_dir)
    parser.add_argument("--fp16",                  action="store_true", default=cfg.fp16)
    parser.add_argument("--bf16",                  action="store_true", default=cfg.bf16)
    parser.add_argument("--smoke_test",            action="store_true", default=False,
                        help="只取512条训练样本+64条验证样本，跑1个epoch，验证流程是否跑通")

    args = parser.parse_args()

    # 将 argparse 结果写回 dataclass
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─────────────────────────── 数据处理 ───────────────────────────
def load_and_tokenize(cfg: TrainingConfig, tokenizer):
    """加载 SST-2 并分词"""
    if cfg.dataset_path:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(cfg.dataset_path, "train.tsv"),
                "validation": os.path.join(cfg.dataset_path, "dev.tsv"),
            },
            delimiter="\t",
        )
    else:
        dataset = load_dataset("glue", cfg.dataset_name)
    train_dataset = dataset["train"]
    eval_dataset  = dataset["validation"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_seq_length,
        )

    train_dataset = train_dataset.map(
        tokenize_fn, batched=True,
        remove_columns=[name for name in ("sentence", "idx") if name in train_dataset.column_names],
        desc="Tokenizing train",
    )
    eval_dataset = eval_dataset.map(
        tokenize_fn, batched=True,
        remove_columns=[name for name in ("sentence", "idx") if name in eval_dataset.column_names],
        desc="Tokenizing eval",
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

    logger.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([x["input_ids"]      for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels":         torch.tensor([x["label"]          for x in batch], dtype=torch.long),
    }


# ─────────────────────────── 模型构建 ───────────────────────────
def build_model(cfg: TrainingConfig):
    """加载 LLaMA2-7B 分类模型，可选 LoRA

    ⚠️  Opacus 限制：per-sample gradient（einsum）要求参数与激活值均为 FP32。
        因此模型必须以 float32 加载，不能使用 torch_dtype=float16/bfloat16。
        AMP autocast 同样需要禁用（见训练循环）。
        A100 80GB 显存足够容纳 7B FP32 + LoRA 梯度，无需降精度。
    """
    logger.info(f"Loading model: {cfg.model_name}")
    logger.info("注意：Opacus 要求 FP32，已忽略 fp16/bf16 设置，模型以 float32 加载")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        torch_dtype=torch.float32,   # Opacus per-sample grad 强制要求 FP32
        device_map=None,             # 手动控制放置，与 Opacus 更兼容
    )

    # LLaMA 没有 pad_token，需手动设置
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    if cfg.use_lora:
        logger.info("Applying LoRA ...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def make_opacus_compatible(model: nn.Module) -> nn.Module:
    """
    Opacus 要求所有层都支持 per-sample gradient。
    fix_and_validate 会自动替换不支持的层（如 BatchNorm → GroupNorm）。
    """
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.warning(f"ModuleValidator found {len(errors)} issue(s), fixing ...")
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=True)  # 确认修复成功
    else:
        logger.info("ModuleValidator: model is Opacus-compatible ✓")
    return model


# ─────────────────────────── 评估 ───────────────────────────
@torch.no_grad()
def evaluate(model, eval_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────── 主训练循环 ───────────────────────────
def train(cfg: TrainingConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── 初始化性能监控器 ──
    profiler = PerformanceProfiler(interval=1.0, output_dir=cfg.output_dir)
    profiler.start()

    # ── Tokenizer ──
    t0 = profiler.begin_stage("tokenizer_load")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    profiler.end_stage("tokenizer_load", t0)

    # ── 数据 ──
    t0 = profiler.begin_stage("data_load_tokenize")
    train_dataset, eval_dataset = load_and_tokenize(cfg, tokenizer)
    profiler.end_stage("data_load_tokenize", t0)

    # smoke_test：截断数据集 + 覆盖超参，几分钟内跑完整条流程
    if cfg.smoke_test:
        train_dataset = train_dataset.select(range(min(512, len(train_dataset))))
        eval_dataset  = eval_dataset.select(range(min(64,  len(eval_dataset))))
        cfg.num_epochs        = 1
        cfg.logical_batch_size = 32
        cfg.per_device_batch_size = 8
        cfg.max_physical_batch_size = 8
        cfg.save_steps  = 9999   # 不保存 checkpoint
        cfg.eval_steps  = 9999   # 不做中间评估
        cfg.logging_steps = 1
        logger.info(
            "⚡ smoke_test 模式：train=512 eval=64 | "
            "logical_bs=32 epochs=1 | 仅验证流程是否跑通"
        )

    # Opacus 接管 DataLoader 后会自动处理 drop_last（DPDataLoader 内部管理），
    # 手动设置 drop_last=True 会触发 "Ignoring drop_last" 警告，改为 False
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.per_device_batch_size * 4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )

    # ── 模型 ──
    t0 = profiler.begin_stage("model_load")
    model = build_model(cfg)
    model = make_opacus_compatible(model)
    model = model.to(device)
    profiler.end_stage("model_load", t0)

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── 学习率调度 ──
    total_steps = (len(train_loader) // (cfg.logical_batch_size // cfg.per_device_batch_size)) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ─────────── Opacus PrivacyEngine ───────────
    # noise_multiplier 由 Opacus 根据 (ε, δ, steps, sample_rate) 自动计算
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=cfg.num_epochs,
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta,
        max_grad_norm=cfg.max_grad_norm,
        # 扩大 alpha 搜索范围，避免 "Optimal order is the largest alpha" 警告
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 512)),
    )

    logger.info(
        f"DP-SGD 配置:\n"
        f"  target ε = {cfg.target_epsilon}, δ = {cfg.target_delta}\n"
        f"  max_grad_norm (C) = {cfg.max_grad_norm}\n"
        f"  noise_multiplier = {optimizer.noise_multiplier:.4f}\n"
        f"  logical_batch_size = {cfg.logical_batch_size}\n"
        f"  physical_batch_size = {cfg.per_device_batch_size}\n"
        f"  total_steps ≈ {total_steps}"
    )

    # AMP scaler — Opacus per-sample gradient (einsum) 要求 FP32 激活，
    # 与 autocast 产生的 FP16/BF16 张量冲突，因此强制禁用。
    # A100 80GB 以 FP32 跑 7B+LoRA 显存充足，无需混合精度。
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # ── 训练 ──
    global_step = 0
    best_acc    = 0.0
    loss_fn     = nn.CrossEntropyLoss()

    try:
        for epoch in range(cfg.num_epochs):
            model.train()
            epoch_loss    = 0.0
            epoch_batches = 0
            epoch_start   = time.time()
            t_epoch       = profiler.begin_stage(f"epoch_{epoch+1}_train")

            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=cfg.max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_loader:

                pbar    = tqdm(memory_safe_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
                step_t0 = time.perf_counter()

                for step, batch in enumerate(pbar):
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels         = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss    = loss_fn(outputs.logits, labels)

                    if not torch.isfinite(loss):
                        print(f"[WARN] Step {step}: loss={loss.item()}, 跳过本 batch", flush=True)
                        optimizer.zero_grad()
                        step_t0 = time.perf_counter()
                        continue

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # 必须在 zero_grad() 前读取，zero_grad() 会将其重置
                    is_logical_step = not optimizer._is_last_step_skipped
                    optimizer.zero_grad()

                    if is_logical_step:
                        scheduler.step()
                        global_step += 1
                        step_elapsed = time.perf_counter() - step_t0
                        profiler.log_step(
                            global_step=global_step,
                            samples_in_step=cfg.logical_batch_size,
                            step_time_s=step_elapsed,
                        )
                        step_t0 = time.perf_counter()

                    epoch_loss    += loss.item()
                    epoch_batches += 1

                    if global_step % cfg.logging_steps == 0 and global_step > 0:
                        eps     = privacy_engine.get_epsilon(cfg.target_delta)
                        recent  = profiler.throughput_log[-cfg.logging_steps:]
                        avg_sps = np.mean([r["throughput_sps"] for r in recent]) if recent else 0.0
                        pbar.set_postfix({
                            "loss":   f"{loss.item():.4f}",
                            "ε":      f"{eps:.2f}",
                            "lr":     f"{scheduler.get_last_lr()[0]:.2e}",
                            "sps":    f"{avg_sps:.1f}",
                            "gpu_MB": f"{profiler.cur_gpu_mb:.0f}",
                        })

                    if global_step % cfg.eval_steps == 0 and global_step > 0:
                        t_eval  = profiler.begin_stage("evaluation")
                        metrics = evaluate(model, eval_loader, device)
                        profiler.end_stage("evaluation", t_eval)
                        eps = privacy_engine.get_epsilon(cfg.target_delta)
                        print(
                            f"[Step {global_step}] acc={metrics['accuracy']:.4f} | "
                            f"f1={metrics['f1']:.4f} | ε={eps:.4f} | "
                            f"gpu_peak={profiler.peak_gpu_mb:.0f}MB | "
                            f"avg_sps={profiler.avg_throughput():.1f}",
                            flush=True,
                        )
                        if metrics["accuracy"] > best_acc:
                            best_acc  = metrics["accuracy"]
                            save_path = os.path.join(cfg.output_dir, "best_model")
                            model._module.save_pretrained(save_path)  # type: ignore
                            tokenizer.save_pretrained(save_path)
                            print(f"  ✓ Best model saved (acc={best_acc:.4f})", flush=True)
                        model.train()

                    if global_step % cfg.save_steps == 0 and global_step > 0:
                        ckpt_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        model._module.save_pretrained(ckpt_path)  # type: ignore
                        tokenizer.save_pretrained(ckpt_path)

            # ── epoch 完成 ──
            profiler.end_stage(f"epoch_{epoch+1}_train", t_epoch)
            epoch_time = time.time() - epoch_start
            eps = privacy_engine.get_epsilon(cfg.target_delta)
            epoch_sps = (
                np.mean([r["throughput_sps"] for r in profiler.throughput_log
                         if r["global_step"] > (epoch * total_steps // cfg.num_epochs)])
                if profiler.throughput_log else 0.0
            )
            avg_loss = epoch_loss / max(1, epoch_batches)
            print(
                f"Epoch {epoch+1} 完成 | avg_loss={avg_loss:.4f} | "
                f"elapsed={epoch_time/60:.1f}min | ε={eps:.4f} | "
                f"epoch_sps={epoch_sps:.1f} samples/sec",
                flush=True,
            )

        # ── 最终评估 ──
        print("=" * 60, flush=True)
        t_eval = profiler.begin_stage("final_evaluation")
        final_metrics = evaluate(model, eval_loader, device)
        profiler.end_stage("final_evaluation", t_eval)
        final_eps = privacy_engine.get_epsilon(cfg.target_delta)
        print(
            f"最终结果:\n"
            f"  Accuracy : {final_metrics['accuracy']:.4f}\n"
            f"  F1       : {final_metrics['f1']:.4f}\n"
            f"  ε (final): {final_eps:.4f}\n"
            f"  δ        : {cfg.target_delta}\n"
            f"  Best acc : {best_acc:.4f}",
            flush=True,
        )
        print("=" * 60, flush=True)

        # 保存最终模型
        final_path = os.path.join(cfg.output_dir, "final_model")
        model._module.save_pretrained(final_path)  # type: ignore
        tokenizer.save_pretrained(final_path)
        print(f"最终模型已保存至: {final_path}", flush=True)

        # ── 性能汇总 ──
        profiler.stop()
        s   = profiler.summary()
        sep = "=" * 64
        print("", flush=True)
        print(sep, flush=True)
        print("  微调完成 — 性能汇总", flush=True)
        print(sep, flush=True)
        print(f"  总运行时间   : {s['wall_time_hms']}  ({s['wall_time_s']:.1f} s)", flush=True)
        print(f"  GPU 显存峰值 : {s['peak_gpu_mb']:.0f} MB  ({s['peak_gpu_mb']/1024:.2f} GB)", flush=True)
        print(f"  CPU 内存峰值 : {s['peak_cpu_rss_mb']:.0f} MB  ({s['peak_cpu_rss_mb']/1024:.2f} GB)", flush=True)
        print(f"  整体吞吐率   : {s['overall_throughput']:.2f} samples/sec", flush=True)
        print(f"  平均步吞吐率 : {s['avg_step_throughput']:.2f} samples/sec", flush=True)
        print(f"  峰值步吞吐率 : {s['peak_step_throughput']:.2f} samples/sec", flush=True)
        print(f"  处理样本总数 : {s['total_samples']:,}", flush=True)
        print(sep, flush=True)
        print("", flush=True)
        profiler.save_reports()

    except Exception:
        import traceback, sys
        print("\n[TRAIN ERROR] 训练过程发生异常：", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


# ─────────────────────────── 入口 ───────────────────────────
if __name__ == "__main__":
    cfg = parse_args()

    logger.info("=" * 60)
    logger.info("DP-SGD Fine-tuning: LLaMA2-7B → SST-2")
    logger.info(f"  model            : {cfg.model_name}")
    logger.info(f"  use_lora         : {cfg.use_lora}  (r={cfg.lora_r})")
    logger.info(f"  target_epsilon   : {cfg.target_epsilon}")
    logger.info(f"  logical_batch_sz : {cfg.logical_batch_size}")
    logger.info(f"  num_epochs       : {cfg.num_epochs}")
    logger.info(f"  fp16/bf16        : fp16={cfg.fp16}, bf16={cfg.bf16}")
    logger.info("=" * 60)

    train(cfg)
