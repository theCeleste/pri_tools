# ============================================================
# FM9G / 9G7B_MHA + LoRA
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

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from peft import PeftModel


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Prompt-aware 低温衰减
# ============================================================

class PromptAwareLowTempWarper(LogitsProcessor):
    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

        self.mapping = {
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

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]

        gen_step = max(1, cur_len - self.prompt_len + 1)
        temp = self.mapping.get(gen_step, 0.50)

        scores = scores.float()
        scores = torch.nan_to_num(
            scores,
            nan=-1e4,
            posinf=1e4,
            neginf=-1e4,
        )

        scores = scores / temp
        scores = torch.clamp(scores, min=-100.0, max=100.0)

        return scores


# ============================================================
# 2. 多数据集提示池构建
# ============================================================

def build_dataset_prompts(batch_size, tokenizer, dataset_name="wikitext"):
    """
    根据数据集名称构建提示
    dataset_name: "sst2", "agnews", "wikitext"
    """
    bos = tokenizer.bos_token or tokenizer.eos_token or ""

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
# 3. 加载 tokenizer
# ============================================================

def load_tokenizer_for_causal_lm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    return tokenizer


# ============================================================
# 4. 加载 9G 模型 + LoRA
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
    print("Auto dtype: using fp32 for numerical stability.")
    return torch.float32


def load_model_with_lora(
    base_model_path,
    lora_path,
    device,
    dtype_mode="auto",
    load_4bit=False,
):
    print(f"Loading base model from: {base_model_path}")

    dtype = select_dtype(dtype_mode)

    if load_4bit:
        print("Loading base model in 4bit mode.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)

    base_model.config.use_cache = False

    if not lora_path or lora_path == "None" or not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA path is required for direct evaluation: {lora_path}")
    if lora_path and lora_path != "None" and os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(base_model, lora_path)
        print("LoRA loaded.")
    else:
        print("No LoRA found, using base model only.")
        model = base_model

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False

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
# 5. 困惑度计算
# ============================================================

def calculate_perplexity(input_sentence, model, tokenizer, device, max_length=512):
    if not input_sentence or not input_sentence.strip():
        return torch.tensor(float("inf")).to(device)

    tokenized = tokenizer(
        input_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    if tokenized.input_ids.size(1) == 0:
        return torch.tensor(float("inf")).to(device)

    input_ids = tokenized.input_ids.to(device, dtype=torch.long)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )

    loss = output.loss

    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(float("inf")).to(device)

    return torch.exp(loss)


# ============================================================
# 6. 保存函数
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
# 7. 确保目录存在
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
# 8. 主函数：生成1000条 + PPL/Zlib排序取前100
# ============================================================

def main(args):
    print("Loading FM9G / 9G model with LoRA support...")

    tokenizer = load_tokenizer_for_causal_lm(args.tokenizer_path or args.base_model_path)

    print("\n=== Loading Model ===")
    model = load_model_with_lora(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        device=device,
        dtype_mode=args.dtype,
        load_4bit=args.load_4bit,
    )

    seq_len = args.seq_len
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
    print(f"Max new tokens: {seq_len}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Output files:")
    print(f"  All samples: {outfile_all}")
    print(f"  Top 100: {outfile_top}")
    print(f"  JSON: {outfile_json}")
    print("=" * 80)

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

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)

            prompt_len = inputs.input_ids.shape[-1]

            logits_warper = LogitsProcessorList([
                PromptAwareLowTempWarper(prompt_len=prompt_len)
            ])

            with torch.no_grad():
                try:
                    if args.decode_mode == "greedy":
                        outputs = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=seq_len,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=False,
                        )
                    else:
                        outputs = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=seq_len,
                            do_sample=True,
                            logits_processor=logits_warper,
                            renormalize_logits=True,
                            temperature=1.0,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=False,
                        )

                except RuntimeError as e:
                    if "probability tensor" in str(e) or "CUDA error" in str(e):
                        print("\nSampling crashed, fallback to greedy decoding...")
                        outputs = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=seq_len,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=False,
                        )
                    else:
                        raise

            texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

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

    print(f"\nTotal valid samples generated: {len(all_samples_data)}")

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
# 9. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FM9G / 9G7B_MHA + LoRA - Multi-dataset generation with PPL/ZLIB sorting"
    )

    parser.add_argument(
        "--base_model_path",
        default="/home/wym/projects/ExtractingModel/Extractingcode/9G7B_MHA",
        type=str,
    )

    parser.add_argument("--tokenizer_path", default="", type=str)

    parser.add_argument(
        "--lora_path",
        default="/home/wym/projects/ExtractingModel/Extractingcode/finetune/train/output_fm9g/ag_news_class0_1_memorize_stable_lr1e4/final_checkpoint",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="agnews",
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
        default=10,
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
        "--decode_mode",
        default="sample",
        choices=["sample", "greedy"],
        type=str,
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        type=str,
    )

    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Use 4bit loading if GPU memory is insufficient",
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
        default="./save/results/9g_agnews",
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
    print(f"  Max New Tokens: {args.seq_len}")
    print(f"  Decode Mode: {args.decode_mode}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    print(f"  dtype: {args.dtype}")
    print(f"  load_4bit: {args.load_4bit}")
    print("=" * 80)

    main(args)
