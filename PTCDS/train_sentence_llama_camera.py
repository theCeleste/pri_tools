# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers LLaMA model for sequence classification with privacy protection."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import sys
import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import ClassLabel
from datasets import Features, Value
import pandas as pd
from datasets import Dataset
from datasets import DatasetDict

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaConfig,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import os

import matplotlib.pyplot as plt
import psutil
import gc

def word_filter(eval_label, filter_list):
    allow_token_ids = (eval_label == filter_list[0])
    for item in filter_list:
        allow_token_ids = allow_token_ids | (eval_label == item)
    return allow_token_ids
    

def evaluate_with_knn_attack(model, dataloader, metric, accelerator, filter_tokens=None, topk=5, target_layer=16):
    """
    评估函数，适配LLaMA模型
    注意：LLaMA使用不同的嵌入层访问方式
    """
    # LLaMA模型的嵌入层访问方式
    emb = model.model.model.embed_tokens.weight
    model.eval()
    samples_seen = 0
    hit_cnt = 0
    total_cnt = 0
    
    for step, batch in enumerate(dataloader):
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = model(**batch)
        
        # evaluate
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        
        attention_mask = batch['attention_mask']
        valid_ids = attention_mask!=0

        eval_label = batch['input_ids']
        valid_ids[word_filter(eval_label, filter_tokens)] = False
        eval_label = eval_label[valid_ids] # (samples)
        
        # 获取目标层的隐藏状态
        if outputs.hidden_states is not None:
            preds_feature = outputs.hidden_states[target_layer][valid_ids]
            ed = torch.cdist(preds_feature, emb, p=2.0) # (samples, embeddings)
            candidate_token_ids_topk = torch.topk(ed, topk, largest=False)[1] # (samples, topk)
            
            hit_cnt += (eval_label.unsqueeze(1) == candidate_token_ids_topk).int().sum().item()
            total_cnt += eval_label.shape[0]
        
    eval_metric = metric.compute()
    if total_cnt > 0:
        eval_metric['knn_top{}'.format(topk)] = hit_cnt/total_cnt
    else:
        eval_metric['knn_top{}'.format(topk)] = 0.0
    return eval_metric


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# 任务到键的映射（保持与原版相同）
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a LLaMA model on a text classification task with privacy protection")
    parser.add_argument(
        "--task_name",
        type=str,
        default='sst2',
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='/home/wpj/model/llama2-7b',  # 默认使用LLaMA-2-7B
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,  # LLaMA模型较大，减小batch size
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,  # LLaMA通常使用较小的学习率
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # 增加梯度累积以处理较大的模型
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Linear LR warmup ratio over total training steps"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        default=0,
        type=int,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    # 隐私保护相关参数
    parser.add_argument(
        "--target_layer",
        type=int,
        default=16  # LLaMA-7B有32层，默认在中间层添加噪声
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--w_noise",
        type=float,
        default=0.1
    )
    # CenterLoss相关参数
    parser.add_argument(
        "--w_cluster_away",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--w_cluster_close",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--recluster",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--cluster_subword",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--cluster_num",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--target_break",
        default=0,
        type=int,
        help="Whether to break after target layer"
    )
    parser.add_argument(
        "--data_sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of data to sample for training (e.g., 0.1 for 10% of data).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use 16-bit (mixed) precision instead of 32-bit.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 (mixed) precision instead of 32-bit.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for efficient fine-tuning.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank for LoRA adaptation.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Alpha parameter for LoRA scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LoRA layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA adaptation.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., 0, 1, 2). If not specified, will use all available GPUs or CUDA_VISIBLE_DEVICES environment variable."
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES environment variable value (e.g., '0,1' or '0'). This will override --gpu_id if both are set."
    )
    
    args = parser.parse_args()
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    
    # 设置GPU设备（必须在导入torch之后，但在初始化Accelerator之前）
    if args.cuda_visible_devices is not None:
        # 如果指定了CUDA_VISIBLE_DEVICES，优先使用它
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        if args.gpu_id is not None:
            print(f"Warning: --cuda_visible_devices={args.cuda_visible_devices} is set, ignoring --gpu_id={args.gpu_id}")
    elif args.gpu_id is not None:
        # 如果只指定了gpu_id，设置CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu_id} to use GPU {args.gpu_id}")
    
    send_example_telemetry("run_llama_privacy_classification", args)
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # 验证GPU设置（在logging初始化之后）
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if args.gpu_id is not None or args.cuda_visible_devices is not None:
            # 当设置了CUDA_VISIBLE_DEVICES后，可见的GPU会重新编号，device 0就是指定的GPU
            logger.info(f"CUDA available: {device_count} GPU(s) visible")
            physical_gpu = args.gpu_id if args.gpu_id is not None else args.cuda_visible_devices
            logger.info(f"Using device: cuda:0 (which corresponds to physical GPU {physical_gpu})")
        else:
            logger.info(f"CUDA available: {device_count} GPU(s), using all available GPUs")
    else:
        logger.warning("CUDA is not available, using CPU")
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 加载数据集
    if args.task_name is not None:
        if args.task_name.lower() == "sst2":
            raw_datasets = load_dataset(
                "csv",
                data_files={
                    "train": "/home/wpj/ptcds/PTCDS/output_PTCDS/SST-2_glove_eps4/eps_4.00/sword_0.50_p_0.20/train.tsv",
                    "validation": "/home/wpj/ptcds/PTCDS/data/SST-2/train.tsv",
                },
                delimiter="\t",
            )
            raw_datasets = raw_datasets.cast_column("label", ClassLabel(names=["negative", "positive"]))
        elif args.task_name in task_to_keys:
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raw_datasets = load_dataset(args.task_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # 数据采样：如果指定了采样比例，则从数据集中抽取指定比例的数据
    if args.data_sample_ratio < 1.0:
        logger.info(f"Sampling {args.data_sample_ratio*100:.1f}% of the dataset...")
        
        # 对训练集进行采样
        if "train" in raw_datasets:
            train_size = int(len(raw_datasets["train"]) * args.data_sample_ratio)
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=args.seed).select(range(train_size))
            logger.info(f"Training set size after sampling: {len(raw_datasets['train'])}")
        
        # 对验证集进行采样
        if "validation" in raw_datasets:
            val_size = int(len(raw_datasets["validation"]) * args.data_sample_ratio)
            raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=args.seed).select(range(val_size))
            logger.info(f"Validation set size after sampling: {len(raw_datasets['validation'])}")
        
        # 对测试集进行采样（如果存在）
        if "test" in raw_datasets:
            test_size = int(len(raw_datasets["test"]) * args.data_sample_ratio)
            raw_datasets["test"] = raw_datasets["test"].shuffle(seed=args.seed).select(range(test_size))
            logger.info(f"Test set size after sampling: {len(raw_datasets['test'])}")

    # 确定标签数量
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # 配置和加载LLaMA模型
    config = LlamaConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels,
        problem_type="single_label_classification" if not is_regression else "regression"
    )
    
    # 添加隐私保护相关配置
    from models.modeling_llama_privacy import add_privacy_config
    config = add_privacy_config(config)
    
    # 设置隐私参数
    config.cluster_num = args.cluster_num
    config.target_layer = args.target_layer
    config.eps = args.eps
    config.w_noise = args.w_noise
    config.w_cluster_away = args.w_cluster_away
    config.w_cluster_close = args.w_cluster_close
    config.recluster = args.recluster
    config.target_break = args.target_break
    config.learning_rate = args.learning_rate
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=not args.use_slow_tokenizer,
        model_input_names=["input_ids", "attention_mask", "cluster_ids"]
    )
    
    # LLaMA tokenizer特殊处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
    
    # 加载隐私保护版本的LLaMA模型
    from models.modeling_llama_privacy import LlamaForSequenceClassification
    
    # 初始化模型（注意：可能需要从头训练分类头）
    model = LlamaForSequenceClassification(config)
    
    # 启用LoRA微调
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA adaptation enabled for efficient fine-tuning.")
    
    # 启用梯度检查点以节省内存
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled to save memory.")
    
    # 如果需要，可以加载预训练的LLaMA权重（只加载基础模型部分）
    if not args.ignore_mismatched_sizes:
        try:
            from transformers import LlamaModel
            pretrained_model = LlamaModel.from_pretrained(args.model_name_or_path)
            model.model.load_state_dict(pretrained_model.state_dict(), strict=False)
            logger.info("Loaded pretrained LLaMA weights (base model only)")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            logger.info("Training from scratch")

    # 数据预处理
    if args.task_name is not None:
        if args.task_name in task_to_keys:
            sentence1_key, sentence2_key = task_to_keys[args.task_name]
        else:
            sentence1_key, sentence2_key = ('text', None)
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # 标签映射
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )

    train_dataset = processed_datasets["train"]
    if args.task_name in task_to_keys:
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    else:
        eval_dataset = processed_datasets["test"] if "test" in processed_datasets else processed_datasets["validation"]

    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 学习率调度器
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    args.num_warmup_steps = int(args.warmup_ratio * args.max_train_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 准备加速器
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 重新计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 检查点设置
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # 初始化跟踪器
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("llama_privacy_classification", experiment_config)

    # 加载评估指标
    class SimpleAccuracy:
        def __init__(self):
            self.correct = 0
            self.total = 0

        def add_batch(self, predictions=None, references=None, **kwargs):
            self.correct += (predictions == references).sum().item()
            self.total += references.numel()

        def compute(self):
            return {"accuracy": (self.correct / self.total) if self.total > 0 else 0.0}

        def reset(self):
            self.correct = 0
            self.total = 0

    metric = SimpleAccuracy()

    # 训练信息
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0
    
    # 训练参数
    knn_topk = 10
    previous_metric = 0
    tolerate = 10
    early_stop = tolerate
    privacy_target = {'sst2': 0.012, 'ag_news': 0.013}
    best_performance = 0
    
    import time
    from cluster_utils_llama import run_cluster, redivide_cluster
    
    # 获取特殊token
    special_tokens = []
    if hasattr(tokenizer, 'special_tokens_map'):
        special_tokens = tokenizer.convert_tokens_to_ids(list(tokenizer.special_tokens_map.values()))
    else:
        # LLaMA的特殊token
        special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        special_tokens = [t for t in special_tokens if t is not None]
    
    filter_tokens = list(set(special_tokens))
    
    # 标签相关词汇（与原版保持一致）
    if args.task_name == 'sst2':
        label_related_words = [['Ġpoorly', 'Ġpointless', 'Ġtires', 'Ġunnecessary', 'Ġbadly'], 
                              ['beaut', 'ple', 'Ġplayful', 'Ġbreathtaking', 'eng']]
    elif args.task_name == 'ag_news':
        label_related_words = [['stocks', 'aspx', '=/', 'http'], 
                              ['ĠMicrosoft', 'Ġsoftware', '=', 'ĠiPod'],
                              ['Ġquarterback', 'Sports', 'Ġtouchdown'],
                              ['Ġmilitants', 'ĠAfghan', 'ĠHamas']]

    # 初始化绘图数据
    plot_interval = 50  # 每多少步绘图一次
    task_losses = []
    privacy_losses = []
    steps_list = []
    
    # 初始化性能监控数据
    batch_stats = []  # 存储每个batch的统计信息
    process = psutil.Process(os.getpid())  # 获取当前进程


    # 训练循环
# 训练循环
    logger.info(">>> entering training loop")
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.recluster and epoch % 5 == 0:
            logger.info('Clustering...')
            model.config.w_noise = 0
            model.config.target_break = 1
            start_time = time.time()
            token2cluster, cluster_center, token_embeddings, sample2vocab = run_cluster(
                model, train_dataloader, tokenizer, label_related_words,
                config.target_layer, config.cluster_num,
                cluster_method='kmeans',
                use_subword=True,
                model_type='llama'
            )
            token2cluster, cluster_center = redivide_cluster(
                label_related_words=label_related_words,
                cluster_center=cluster_center,
                tokenizer=tokenizer,
                token2cluster=token2cluster,
                token_embeddings=token_embeddings,
                embeddings_to_vocabulary=sample2vocab,
                cluster_num=config.cluster_num
            )
            model.privacy_loss.cluster_embedding.weight.data = cluster_center.type_as(
                model.privacy_loss.cluster_embedding.weight.data
            )
            model.config.w_noise = args.w_noise
            model.config.target_break = 0
            logger.info(f'Clustering done! Cost {time.time()-start_time:.2f} seconds')

        if early_stop == 0:
            break
        model.train()
        total_loss = 0
        total_task_loss = 0
        total_privacy_loss = 0
        epoch_batch_stats = []

        # ✅ batch 级进度条：放在 epoch 循环里面
        train_iter = tqdm(
            train_dataloader,
            desc=f"Train Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            file=sys.stderr,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(train_iter):
            if step == 0:
                logger.info(">>> got first batch, starting forward")
            batch_start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**3
                memory_reserved_before = torch.cuda.memory_reserved() / 1024**3
            else:
                memory_before = process.memory_info().rss / 1024**3
                memory_reserved_before = 0

            batch_size = batch['input_ids'].shape[0]

            if config.recluster and 'token2cluster' in locals():
                default_cluster_id = 0
                batch['cluster_ids'] = torch.tensor(
                    [[token2cluster.get(ids.item(), default_cluster_id) for ids in batch_ids]
                    for batch_ids in batch['input_ids']],
                    device=batch['input_ids'].device
                )

            #outputs = model(**batch, use_cache=False, output_hidden_states=True)
            outputs = model(**batch)
            task_loss = outputs.loss
            privacy_loss_dict = outputs.privacy_loss
            privacy_loss = 0
            if privacy_loss_dict is not None:
                for loss_item in privacy_loss_dict.values():
                    privacy_loss += loss_item

            loss = task_loss + privacy_loss

            # ✅ 让进度条显示 loss
            if accelerator.is_local_main_process:
                train_iter.set_postfix(
                    task=float(task_loss.detach().cpu()),
                    pri=float(privacy_loss.detach().cpu()) if isinstance(privacy_loss, torch.Tensor) else float(privacy_loss),
                    opt_step=completed_steps,
                )

            total_loss += loss.detach().float()
            total_task_loss += task_loss.detach().float()
            total_privacy_loss += privacy_loss.detach().float() if isinstance(privacy_loss, torch.Tensor) else privacy_loss

            task_losses.append(task_loss.detach().cpu().item())
            privacy_losses.append(privacy_loss.detach().cpu().item() if isinstance(privacy_loss, torch.Tensor) else privacy_loss)
            steps_list.append(completed_steps)

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            # ========== 结束性能监控 ==========
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保CUDA操作完成
                memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved_after = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            else:
                memory_after = process.memory_info().rss / 1024**3  # GB
                memory_reserved_after = 0
                memory_peak = memory_after
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # 计算吞吐率（样本数/秒）
            throughput = batch_size / batch_time if batch_time > 0 else 0
            
            # 记录统计信息
            batch_stat = {
                'epoch': epoch,
                'step': step,
                'completed_steps': completed_steps,
                'batch_size': batch_size,
                'batch_time': batch_time,
                'throughput': throughput,  # 样本/秒
                'memory_before_gb': memory_before,
                'memory_after_gb': memory_after,
                'memory_reserved_before_gb': memory_reserved_before,
                'memory_reserved_after_gb': memory_reserved_after,
                'memory_peak_gb': memory_peak,
                'memory_delta_gb': memory_after - memory_before,
            }
            batch_stats.append(batch_stat)  # 累积所有batch的统计信息
            epoch_batch_stats.append(batch_stat)  # 当前epoch的统计信息
            
            # 输出统计信息
            if accelerator.is_local_main_process:
                logger.info(
                    f"Epoch {epoch}, Step {step} (Completed: {completed_steps}): "
                    f"Batch Size={batch_size}, "
                    f"Time={batch_time:.4f}s, "
                    f"Throughput={throughput:.2f} samples/s, "
                    f"Memory={memory_after:.2f}GB (Peak={memory_peak:.2f}GB, Delta={batch_stat['memory_delta_gb']:+.2f}GB)"
                )
            # ========== 性能监控结束 ==========

            # 动态绘图
            if completed_steps % plot_interval == 0 and completed_steps > 0:
                # Task Loss
                plt.figure()
                plt.plot(steps_list, task_losses, label='Task Loss', color='blue')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title('Task Loss Curve')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, 'task_loss_curve.png'))
                plt.close()
                
                # Privacy Loss
                plt.figure()
                plt.plot(steps_list, privacy_losses, label='Privacy Loss', color='red')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title('Privacy Loss Curve')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, 'privacy_loss_curve.png'))
                plt.close()
        if hasattr(metric, "reset"):
            metric.reset()
        # 评估
        eval_metric = evaluate_with_knn_attack(
            model, eval_dataloader, metric, accelerator, 
            target_layer=args.target_layer, topk=knn_topk, filter_tokens=filter_tokens
        )
        
        # 保存最佳模型
        if args.task_name in privacy_target and eval_metric[f'knn_top{knn_topk}'] <= privacy_target[args.task_name]:
            if eval_metric['accuracy'] > best_performance:
                best_performance = eval_metric['accuracy']
                
                output_dir = f'{args.output_dir}/epoch{epoch}'
                os.makedirs(output_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                    json.dump({"eval_accuracy": eval_metric["accuracy"], "privacy_score": eval_metric[f'knn_top{knn_topk}']}, f)
        
        # Early stopping
        if previous_metric == eval_metric['accuracy']:
            early_stop -= 1
        else:
            early_stop = tolerate
        previous_metric = eval_metric['accuracy']

        logger.info(f"Epoch {epoch}: {eval_metric}")
        progress_bar.set_description(f'acc:{eval_metric["accuracy"]:.4f}')
        
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric["accuracy"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
        
        # 每个epoch结束后保存性能统计信息
        if accelerator.is_main_process and args.output_dir is not None:
            stats_file = os.path.join(args.output_dir, f'batch_stats_epoch_{epoch}.json')
            with open(stats_file, 'w') as f:
                json.dump(epoch_batch_stats, f, indent=2)
            logger.info(f"Epoch {epoch} batch statistics saved to {stats_file}")
            
            # 计算并输出epoch级别的汇总统计
            if epoch_batch_stats:
                epoch_stats = {
                    'epoch': epoch,
                    'total_batches': len(epoch_batch_stats),
                    'avg_batch_time': sum(s['batch_time'] for s in epoch_batch_stats) / len(epoch_batch_stats),
                    'avg_throughput': sum(s['throughput'] for s in epoch_batch_stats) / len(epoch_batch_stats),
                    'total_samples': sum(s['batch_size'] for s in epoch_batch_stats),
                    'total_time': sum(s['batch_time'] for s in epoch_batch_stats),
                    'avg_memory_gb': sum(s['memory_after_gb'] for s in epoch_batch_stats) / len(epoch_batch_stats),
                    'max_memory_gb': max(s['memory_peak_gb'] for s in epoch_batch_stats),
                    'min_memory_gb': min(s['memory_before_gb'] for s in epoch_batch_stats),
                }
                logger.info(
                    f"Epoch {epoch} Summary: "
                    f"Avg Time={epoch_stats['avg_batch_time']:.4f}s/batch, "
                    f"Avg Throughput={epoch_stats['avg_throughput']:.2f} samples/s, "
                    f"Total Samples={epoch_stats['total_samples']}, "
                    f"Total Time={epoch_stats['total_time']:.2f}s, "
                    f"Avg Memory={epoch_stats['avg_memory_gb']:.2f}GB, "
                    f"Peak Memory={epoch_stats['max_memory_gb']:.2f}GB"
                )
                
                # 保存epoch汇总统计
                epoch_stats_file = os.path.join(args.output_dir, f'epoch_stats_epoch_{epoch}.json')
                with open(epoch_stats_file, 'w') as f:
                    json.dump(epoch_stats, f, indent=2)
    
    # 训练结束
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
        
        # 保存所有batch的统计信息到单个文件
        if accelerator.is_main_process and batch_stats:
            all_stats_file = os.path.join(args.output_dir, 'all_batch_stats.json')
            with open(all_stats_file, 'w') as f:
                json.dump(batch_stats, f, indent=2)
            logger.info(f"All batch statistics saved to {all_stats_file}")
            
            # 计算并保存总体统计信息
            if batch_stats:
                overall_stats = {
                    'total_batches': len(batch_stats),
                    'total_epochs': args.num_train_epochs,
                    'avg_batch_time': sum(s['batch_time'] for s in batch_stats) / len(batch_stats),
                    'avg_throughput': sum(s['throughput'] for s in batch_stats) / len(batch_stats),
                    'total_samples': sum(s['batch_size'] for s in batch_stats),
                    'total_time': sum(s['batch_time'] for s in batch_stats),
                    'avg_memory_gb': sum(s['memory_after_gb'] for s in batch_stats) / len(batch_stats),
                    'max_memory_gb': max(s['memory_peak_gb'] for s in batch_stats),
                    'min_memory_gb': min(s['memory_before_gb'] for s in batch_stats),
                }
                overall_stats_file = os.path.join(args.output_dir, 'overall_stats.json')
                with open(overall_stats_file, 'w') as f:
                    json.dump(overall_stats, f, indent=2)
                logger.info(f"Overall statistics saved to {overall_stats_file}")
                logger.info(
                    f"Training Complete - Overall: "
                    f"Avg Throughput={overall_stats['avg_throughput']:.2f} samples/s, "
                    f"Total Samples={overall_stats['total_samples']}, "
                    f"Total Time={overall_stats['total_time']:.2f}s, "
                    f"Peak Memory={overall_stats['max_memory_gb']:.2f}GB"
                )


if __name__ == "__main__":
    main()
"""
CUDA_VISIBLE_DEVICES=1 python ./train_sentence_llama_camera.py   --model_name_or_path /home/wpj/model/llama2-7b   --output_dir /home/wpj/ptcds/PTCDS/outputs/llama2_sst2_run1   --task_name sst2   --per_device_train_batch_size 16   --per_device_eval_batch_size 16   --gradient_accumulation_steps 4   --num_train_epochs 1   --learning_rate 1e-4   --target_layer 16   --eps 1.0   --cluster_num 100   --w_noise 0.1   --w_cluster_close 0.1   --w_cluster_away 0.05   --data_sample_ratio 1.0   --use_lora   --lora_rank 16   --lora_alpha 16   --lora_dropout 0.1   --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj   --bf16   --max_grad_norm 3.0   --warmup_ratio 0.05   --max_length 128

"""