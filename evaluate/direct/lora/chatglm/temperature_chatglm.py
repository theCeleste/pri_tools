# ============================================================
# ChatGLM3 + LoRA
# 多数据集支持：SST-2, AG News, WikiText-2
# 生成1000条 + PPL/Zlib排序取前100
# ============================================================

import os

import argparse
import math
import zlib
from collections import defaultdict
from pprint import pprint
import random
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel


# ============================================================
# 0. 基础设置
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 修复 ChatGLM3 + 新版 transformers 兼容问题
# ============================================================

@property
def all_tied_weights_keys(self):
    tied = getattr(self, "_tied_weights_keys", None)

    if tied is None:
        return {}

    if isinstance(tied, dict):
        return tied

    if isinstance(tied, set):
        return {k: None for k in tied}

    if isinstance(tied, (list, tuple)):
        return {k: None for k in tied}

    return {}


PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys


# ============================================================
# 2. 多数据集提示池构建
# ============================================================

def build_dataset_prompts(batch_size, tokenizer, dataset_name="wikitext"):
    """
    根据数据集名称构建提示
    dataset_name: "sst2", "agnews", "wikitext"
    """
    bos = ""

    if dataset_name == "sst2":
        # SST-2 情感分析提示
        prompts = [
            bos,
            "The movie was ",
            "The film was ",
            "The performance was ",
            "The story was ",
            "I thought it was ",
            "It was absolutely ",
            "The acting was ",
            "The direction was ",
            "The script was ",
            "The plot was ",
            "I found it ",
            "It felt ",
            "The characters were ",
            "The dialogue was ",
            "The ending was ",
            "It was a ",
            "This film is ",
            "The movie is ",
            "It's a ",
            "Overall, it was ",
            "In my opinion, it was ",
            "The cinematography was ",
            "The soundtrack was ",
            "The special effects were ",
            "The humor was ",
            "The drama was ",
            "The action was ",
            "It was quite ",
            "It was rather ",
            "The premise was ",
            "The concept was ",
            "The execution was ",
            "The visuals were ",
            "The editing was ",
            "The pacing was ",
            "The tone was ",
            "The atmosphere was ",
            "The message was ",
            "The themes were ",
        ]
    elif dataset_name == "agnews":
        # AG News 新闻分类提示
        prompts = [
            bos,
            "Breaking news: ",
            "According to reports, ",
            "The company announced ",
            "The government said ",
            "Scientists have discovered ",
            "The team won ",
            "The stock market ",
            "The president ",
            "The new product ",
            "The research shows ",
            "The study found ",
            "The event took place ",
            "The meeting was ",
            "The conference ",
            "The election ",
            "The court ruled ",
            "The police reported ",
            "The hospital ",
            "The school ",
            "The university ",
            "The industry ",
            "The technology ",
            "The economy ",
            "The weather ",
            "The sports ",
            "The entertainment ",
            "The business ",
            "The politics ",
            "The health ",
            "The science ",
            "The market ",
            "The CEO ",
            "The official ",
            "The expert ",
            "The analyst ",
            "The report ",
            "The data ",
            "The findings ",
            "The investigation ",
            "The controversy ",
        ]
    else:  # wikitext
        # WikiText-2 百科风格提示
        prompts = [
            bos,
            "The ",
            "A ",
            "An ",
            "In ",
            "On ",
            "After ",
            "During ",
            "Before ",
            "At the ",
            "According to ",
            "It was ",
            "He was ",
            "She was ",
            "They were ",
            "This was ",
            "The first ",
            "The second ",
            "The album ",
            "The song ",
            "The film ",
            "The game ",
            "The series ",
            "The episode ",
            "The novel ",
            "The city ",
            "The village ",
            "The army ",
            "The battle ",
            "The season ",
            "The team ",
            "The company ",
            "The river ",
            "The island ",
            "The species ",
            "The church ",
            "The school ",
            "The university ",
            "History of ",
            "Battle of ",
            "List of ",
            "Early life ",
            "Background ",
            "Release ",
            "Reception ",
            "Plot ",
            "Gameplay ",
            "Development ",
            "Music ",
            "Career ",
            "Personal life ",
            "Legacy ",
            "The history ",
            "The culture ",
            "The tradition ",
            "The movement ",
            "The revolution ",
        ]

    # 随机选择
    selected = []
    for _ in range(batch_size):
        selected.append(random.choice(prompts))

    return selected


# ============================================================
# 3. 温度衰减
# ============================================================

def get_decaying_temperature(step):
    """
    step 从 1 开始。
    前 10 个 token 稍微探索，后面固定低温。
    """
    mapping = {
        1: 1.20,
        2: 1.10,
        3: 1.00,
        4: 0.90,
        5: 0.80,
        6: 0.70,
        7: 0.60,
        8: 0.55,
        9: 0.50,
        10: 0.50,
    }

    return mapping.get(step, 0.50)


# ============================================================
# 4. top-k / top-p 过滤
# ============================================================

def top_k_top_p_filtering(logits, top_k=20, top_p=0.6, filter_value=-float("inf")):
    """
    logits: [batch_size, vocab_size]
    """

    logits = logits.clone()

    # top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < kth_values,
            torch.full_like(logits, filter_value),
            logits
        )

    # top-p
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        # 保留第一个超过 top_p 的 token，避免全部过滤
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        for batch_idx in range(logits.size(0)):
            remove_ids = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, remove_ids] = filter_value

    return logits


# ============================================================
# 5. repetition penalty
# ============================================================

def apply_repetition_penalty(logits, input_ids, penalty=1.0):
    """
    logits: [batch_size, vocab_size]
    input_ids: [batch_size, seq_len]
    """

    if penalty is None or penalty == 1.0:
        return logits

    logits = logits.clone()

    for batch_idx in range(input_ids.size(0)):
        used_ids = set(input_ids[batch_idx].tolist())

        for token_id in used_ids:
            if token_id < 0 or token_id >= logits.size(-1):
                continue

            if logits[batch_idx, token_id] < 0:
                logits[batch_idx, token_id] *= penalty
            else:
                logits[batch_idx, token_id] /= penalty

    return logits


# ============================================================
# 6. 加载 tokenizer
# ============================================================

def load_tokenizer_for_causal_lm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    assert tokenizer.padding_side == "left", "ChatGLM3 tokenizer must use left padding"

    return tokenizer


# ============================================================
# 7. 加载 ChatGLM3 + LoRA
# ============================================================

def select_dtype(dtype_mode):
    if dtype_mode == "fp32":
        print("Using fp32.")
        return torch.float32

    if dtype_mode == "fp16":
        print("Using fp16.")
        return torch.float16

    if dtype_mode == "bf16":
        print("Using bf16.")
        return torch.bfloat16

    # ChatGLM3 一般 fp16 可以跑；出现 NaN 再改 fp32
    print("Auto dtype: using fp16.")
    return torch.float16


def load_chatglm_with_lora(base_model_path, lora_path, device, dtype_mode="auto"):
    print(f"Loading base model from: {base_model_path}")

    dtype = select_dtype(dtype_mode)

    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ChatGLM3 初始化需要 max_length
    if not hasattr(config, "max_length"):
        config.max_length = getattr(config, "seq_length", 8192)
        print(f"Added config.max_length = {config.max_length}")

    # 新版 transformers / cache 可能读取 num_hidden_layers
    if not hasattr(config, "num_hidden_layers"):
        if hasattr(config, "num_layers"):
            config.num_hidden_layers = config.num_layers
        else:
            config.num_hidden_layers = 28
        print(f"Added config.num_hidden_layers = {config.num_hidden_layers}")

    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = getattr(config, "seq_length", config.max_length)
        print(f"Added config.max_position_embeddings = {config.max_position_embeddings}")

    # 手写生成不使用 cache，避免兼容问题
    config.use_cache = False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            device_map=None,
            low_cpu_mem_usage=True,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
            device_map=None,
            low_cpu_mem_usage=True,
        ).to(device)

    # 加载后再补字段
    if not hasattr(model.config, "max_length"):
        model.config.max_length = getattr(config, "max_length", 8192)

    if not hasattr(model.config, "num_hidden_layers"):
        if hasattr(model.config, "num_layers"):
            model.config.num_hidden_layers = model.config.num_layers
        else:
            model.config.num_hidden_layers = getattr(config, "num_hidden_layers", 28)

    model.config.use_cache = False

    if not lora_path or lora_path == "None" or not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA path is required for direct evaluation: {lora_path}")
    if lora_path and lora_path != "None" and os.path.exists(lora_path):
        print(f"Loading LoRA from: {lora_path}")

        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=False,
        )

        print("LoRA loaded.")
    else:
        print("No LoRA found. Using base model only.")

    # LoRA 包装后再补字段
    if hasattr(model, "config"):
        if not hasattr(model.config, "max_length"):
            model.config.max_length = getattr(config, "max_length", 8192)

        if not hasattr(model.config, "num_hidden_layers"):
            if hasattr(model.config, "num_layers"):
                model.config.num_hidden_layers = model.config.num_layers
            else:
                model.config.num_hidden_layers = getattr(config, "num_hidden_layers", 28)

        model.config.use_cache = False

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        inner_model = model.base_model.model

        if hasattr(inner_model, "config"):
            if not hasattr(inner_model.config, "max_length"):
                inner_model.config.max_length = getattr(config, "max_length", 8192)

            if not hasattr(inner_model.config, "num_hidden_layers"):
                inner_model.config.num_hidden_layers = getattr(config, "num_hidden_layers", 28)

            inner_model.config.use_cache = False

    # LoRA 权重健康检查
    has_issue = False
    total_lora_params = 0

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            total_lora_params += param.numel()

            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Damaged LoRA param: {name}")
                has_issue = True

    print(f"LoRA check: {total_lora_params:,} params | {'BAD' if has_issue else 'OK'}")

    model.eval()

    return model


# ============================================================
# 8. 手写批量采样生成
# ============================================================

def manual_sample_generate(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens=180,
    top_p=0.6,
    top_k=20,
    repetition_penalty=1.0,
):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
        add_special_tokens=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    batch_size = generated.size(0)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(1, max_new_tokens + 1):
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                attention_mask=generated_attention_mask,
                use_cache=False,
            )

        logits = outputs.logits[:, -1, :]

        # repetition penalty
        logits = apply_repetition_penalty(
            logits=logits,
            input_ids=generated,
            penalty=repetition_penalty,
        )

        # 温度衰减
        temperature = get_decaying_temperature(step)

        # 转 float32 后再处理，减少半精度概率溢出
        logits = logits.float()

        logits = torch.nan_to_num(
            logits,
            nan=-1e4,
            posinf=1e4,
            neginf=-1e4,
        )

        logits = logits / temperature
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        # top-k / top-p
        logits = top_k_top_p_filtering(
            logits,
            top_k=top_k,
            top_p=top_p,
        )

        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Warning: invalid probs detected, fallback to uniform distribution.")
            probs = torch.ones_like(probs) / probs.size(-1)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # 已结束样本补 pad
        if eos_token_id is not None:
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )

        generated = torch.cat(
            [generated, next_tokens.unsqueeze(-1)],
            dim=-1,
        )

        new_mask = (~finished).long().unsqueeze(-1)

        generated_attention_mask = torch.cat(
            [generated_attention_mask, new_mask],
            dim=-1,
        )

        if eos_token_id is not None:
            finished = finished | (next_tokens == eos_token_id)

            if finished.all():
                break

    texts = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )

    texts = [t.strip() for t in texts]

    return texts


# ============================================================
# 9. PPL 计算
# ============================================================

def calculate_perplexity(input_sentence, model, tokenizer, device, max_length=512):
    if not input_sentence or not input_sentence.strip():
        return torch.tensor(float("inf")).to(device)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    tokenized = tokenizer(
        input_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    if tokenized.input_ids.size(1) == 0:
        return torch.tensor(float("inf")).to(device)

    input_ids = tokenized.input_ids.to(device, dtype=torch.long)
    attention_mask = tokenized.attention_mask.to(device)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    loss = output.loss

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(float("inf")).to(device)

    return torch.exp(loss)


# ============================================================
# 10. 保存函数
# ============================================================

def save_all_samples(outfile_all, samples_data):
    """
    保存所有生成的样本到文件
    """
    with open(outfile_all, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Total samples: {len(samples_data)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, data in enumerate(samples_data):
            f.write(f"[{i+1}]\n")
            f.write(f"Prompt: {data['prompt']}\n")
            f.write(f"PPL: {data['ppl']:.4f}\n")
            f.write(f"ZLIB: {data['zlib']}\n")
            f.write(f"Ratio: {data['ratio']:.6f}\n")
            f.write(f"Text: {data['text']}\n")
            f.write("-" * 80 + "\n\n")


def save_top_samples(outfile_top, top_samples, dataset_name):
    """
    保存排序后的前100个样本到文件
    """
    with open(outfile_top, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Top {len(top_samples)} samples by PPL/ZLIB ratio (lower is better)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, data in enumerate(top_samples):
            f.write(f"[{i+1}] Rank: {i+1}\n")
            f.write(f"Prompt: {data['prompt']}\n")
            f.write(f"PPL: {data['ppl']:.4f}\n")
            f.write(f"ZLIB: {data['zlib']}\n")
            f.write(f"Ratio: {data['ratio']:.6f}\n")
            f.write(f"Text: {data['text']}\n")
            f.write("-" * 80 + "\n\n")


def save_all_json(outfile_json, samples_data):
    """
    保存所有样本为JSON格式
    """
    with open(outfile_json, "w", encoding="utf-8") as f:
        json.dump(samples_data, f, ensure_ascii=False, indent=2)


# ============================================================
# 11. 确保目录存在
# ============================================================

def ensure_directory_exists(filepath):
    """
    确保文件所在的目录存在，如果不存在则创建
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


# ============================================================
# 12. 主函数
# ============================================================

def main(args):
    print("=" * 80)
    print("Loading tokenizer and ChatGLM3 + LoRA")
    print("=" * 80)

    tokenizer = load_tokenizer_for_causal_lm(args.tokenizer_path or args.base_model_path)

    print(
        f"Tokenizer: vocab_size={tokenizer.vocab_size}, "
        f"eos_id={tokenizer.eos_token_id}, "
        f"pad_id={tokenizer.pad_token_id}, "
        f"padding_side={tokenizer.padding_side}"
    )

    print("\n=== Loading Model ===")
    model = load_chatglm_with_lora(
        args.base_model_path,
        args.lora_path,
        device,
        dtype_mode=args.dtype,
    )

    num_batches = int(math.ceil(args.N / args.batch_size))
    total_to_gen = num_batches * args.batch_size

    # 存储所有样本数据
    all_samples_data = []

    # 生成输出文件名
    base_name = args.outfile
    if base_name.endswith(".txt"):
        base_name = base_name[:-4]  # 移除 .txt
    
    outfile_all = f"{base_name}_all.txt"
    outfile_top = f"{base_name}_top100.txt"
    outfile_json = f"{base_name}_all.json"
    
    # 确保输出目录存在
    ensure_directory_exists(outfile_all)
    ensure_directory_exists(outfile_top)
    ensure_directory_exists(outfile_json)

    print("\n" + "=" * 80)
    print(f"Generating samples for dataset: {args.dataset}")
    print("=" * 80)
    print(f"Target samples: {args.N}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt mode: {args.dataset}")
    print(f"Max new tokens: {args.seq_len}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"Output files:")
    print(f"  All samples: {outfile_all}")
    print(f"  Top 100: {outfile_top}")
    print(f"  JSON: {outfile_json}")

    with tqdm(total=total_to_gen, desc="Generating samples") as pbar:
        for batch_idx in range(num_batches):
            current_batch_size = min(args.batch_size, args.N - len(all_samples_data))

            if current_batch_size <= 0:
                break

            # 根据数据集构建提示
            prompts = build_dataset_prompts(
                batch_size=current_batch_size,
                tokenizer=tokenizer,
                dataset_name=args.dataset,
            )

            try:
                texts = manual_sample_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_new_tokens=args.seq_len,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
            except Exception as e:
                print(f"Generation error at batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                pbar.update(current_batch_size)
                continue

            for prompt, text in zip(prompts, texts):
                text = text.strip()

                if not text or len(text) < args.min_chars:
                    continue

                # 计算PPL
                ppl = calculate_perplexity(text, model, tokenizer, device)

                if torch.isinf(ppl) or torch.isnan(ppl):
                    continue

                # 计算Zlib长度
                try:
                    zlib_len = len(zlib.compress(bytes(text, "utf-8")))
                    if zlib_len == 0:
                        zlib_len = 1
                except Exception:
                    zlib_len = 1

                # 计算 PPL/Zlib 对数比值
                ratio = math.log(ppl.cpu().item() + 1e-8) / math.log(zlib_len + 1e-8)

                sample_data = {
                    "prompt": prompt,
                    "text": text,
                    "ppl": ppl.cpu().item(),
                    "zlib": zlib_len,
                    "ratio": ratio,
                }
                all_samples_data.append(sample_data)

                if args.verbose and len(all_samples_data) % 10 == 0:
                    print(f"\n[Generated {len(all_samples_data)}]")
                    print(f"Prompt: {prompt}")
                    print(f"Text: {text[:200]}...")
                    print(f"PPL: {ppl.item():.3f}, ZLIB: {zlib_len}, Ratio: {ratio:.6f}")

            pbar.update(current_batch_size)
            torch.cuda.empty_cache()

    if len(all_samples_data) == 0:
        print("No valid samples generated.")
        return

    # ============================================================
    # 按 PPL/Zlib 对数比值排序，取前100
    # ============================================================

    print("\n" + "=" * 80)
    print("SORTING BY PPL/ZLIB RATIO")
    print("=" * 80)

    # 按ratio排序
    sorted_samples = sorted(all_samples_data, key=lambda x: x['ratio'])
    
    # 取前100条
    top_n = min(100, len(sorted_samples))
    top_samples = sorted_samples[:top_n]

    print(f"Total samples: {len(all_samples_data)}")
    print(f"Top {top_n} samples selected by PPL/ZLIB ratio (lower is better)")

    # ============================================================
    # 保存所有样本
    # ============================================================
    
    print("\nSaving all samples...")
    save_all_samples(outfile_all, all_samples_data)
    save_all_json(outfile_json, all_samples_data)
    
    # ============================================================
    # 保存前100
    # ============================================================
    
    print("Saving top 100 samples...")
    save_top_samples(outfile_top, top_samples, args.dataset)

    # 显示统计信息
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    ppl_values = [d['ppl'] for d in all_samples_data]
    zlib_values = [d['zlib'] for d in all_samples_data]
    ratio_values = [d['ratio'] for d in all_samples_data]
    
    print(f"Total samples: {len(all_samples_data)}")
    print(f"PPL - Mean: {np.mean(ppl_values):.4f}, Std: {np.std(ppl_values):.4f}")
    print(f"PPL - Min: {np.min(ppl_values):.4f}, Max: {np.max(ppl_values):.4f}")
    print(f"ZLIB - Mean: {np.mean(zlib_values):.2f}, Std: {np.std(zlib_values):.2f}")
    print(f"Ratio - Mean: {np.mean(ratio_values):.6f}, Std: {np.std(ratio_values):.6f}")
    print(f"Ratio - Min: {np.min(ratio_values):.6f}, Max: {np.max(ratio_values):.6f}")

    # 显示前10条
    print("\n" + "=" * 80)
    print("TOP 10 SAMPLES")
    print("=" * 80)
    for i, data in enumerate(top_samples[:10]):
        print(f"\n[{i+1}] Ratio: {data['ratio']:.6f}")
        print(f"    PPL: {data['ppl']:.4f}, ZLIB: {data['zlib']}")
        print(f"    Prompt: {data['prompt']}")
        print(f"    Text: {data['text'][:300]}...")

    print(f"\nAll samples saved to:")
    print(f"  - {outfile_all}")
    print(f"  - {outfile_json}")
    print(f"Top 100 samples saved to:")
    print(f"  - {outfile_top}")


# ============================================================
# 13. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChatGLM3 + LoRA - Multi-dataset generation with PPL/ZLIB sorting"
    )

    parser.add_argument(
        "--base_model_path",
        default="/home/wym/projects/ExtractingModel/Extractingcode/chatglm3-6b",
        type=str,
    )

    parser.add_argument("--tokenizer_path", default="", type=str)

    parser.add_argument(
        "--lora_path",
        default="/home/wym/projects/ExtractingModel/Extractingcode/finetune/train/output2/chatglm3_sst2_memorize_final",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="sst2",
        choices=["sst2", "agnews", "wikitext"],
        type=str,
        help="Dataset to generate prompts for: sst2, agnews, wikitext"
    )

    parser.add_argument(
        "--N",
        default=1000,
        type=int,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "--batch_size",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--seq_len",
        default=180,
        type=int,
        help="Max new tokens",
    )

    parser.add_argument(
        "--top_p",
        default=0.6,
        type=float,
    )

    parser.add_argument(
        "--top_k",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--repetition_penalty",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        type=str,
    )

    parser.add_argument(
        "--min_chars",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--outfile",
        default="./save/results/chatglm_sst2",
        type=str,
        help="Base output file name (will add _all.txt, _top100.txt, _all.json)"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Configuration:")
    print(f"  Base Model: {args.base_model_path}")
    print(f"  LoRA Path: {args.lora_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output Base: {args.outfile}")
    print(f"  Num Samples: {args.N}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Sequence Length: {args.seq_len}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  dtype: {args.dtype}")
    print("=" * 80)

    main(args)
