#!/usr/bin/env python3
import argparse
import os
import sys
import time
import torch
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

class LoggerTee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):

        return getattr(self.terminal, "isatty", lambda: False)()

def load_single_split(tokenizer, path, split_name, max_len=128):
    """根据路径类型动态加载单个 Split 数据的通用函数"""
    if os.path.isdir(path):
        try:
            full_dataset = load_from_disk(path)
            if isinstance(full_dataset, dict) or hasattr(full_dataset, "keys"):
                dataset = full_dataset[split_name] if split_name in full_dataset else full_dataset[list(full_dataset.keys())[0]]
            else:
                dataset = full_dataset
        except Exception:
            specific_path = os.path.join(path, split_name)
            if os.path.exists(specific_path):
                dataset = load_from_disk(specific_path)
            else:
                ext = None
                for file in os.listdir(path):
                    if split_name in file or (split_name == "validation" and "dev" in file):
                        if file.endswith(".tsv"): ext = "csv"; delimiter = "\t"
                        elif file.endswith(".csv"): ext = "csv"; delimiter = ","
                        elif file.endswith(".json") or file.endswith(".jsonl"): ext = "json"
                        if ext:
                            path = os.path.join(path, file)
                            break
                if ext:
                    kwargs = {"delimiter": delimiter} if ext == "csv" and "delimiter" in locals() else {}
                    dataset = load_dataset(ext, data_files={split_name: path}, split=split_name, **kwargs)
                else:
                    raise FileNotFoundError(f"无法在目录 {path} 中识别出 {split_name} 数据。")
    else:
        if path.endswith(".tsv"):
            dataset = load_dataset("csv", data_files={split_name: path}, delimiter="\t", split=split_name)
        elif path.endswith(".csv"):
            dataset = load_dataset("csv", data_files={split_name: path}, split=split_name)
        elif path.endswith(".json") or path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files={split_name: path}, split=split_name)
        else:
            raise ValueError(f"无法识别的文件格式: {path}")

    dataset = dataset.filter(lambda example: example["label"] != -1)

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return encoded_dataset

def prepare_model(token: str, is_lora: bool = True, lora_rank: int = 16, model_name: str = "meta-llama/Llama-2-7b-hf"):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, torch_dtype=torch.bfloat16, device_map=None, token=token
    )
    print(f"Model {model_name} loaded")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  
        pretrained_model.config.pad_token_id = tokenizer.eos_token_id
        
    if is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(pretrained_model, lora_config)
    else:
        model = pretrained_model

    for name, param in model.named_parameters():
        if "embed_tokens" in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    return model, tokenizer

def train_step(model, optimizer, criterion, batch, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()
    acc = correct / total
    model.train()
    return acc

def save_model(model, tokenizer, base_dir, suffix, time_str):
    # 使用训练开始时统一生成的时间戳，确保模型和 log 文件夹名字完全对齐
    final_path = os.path.join(base_dir, f"{suffix}_{time_str}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

def train(args, device):
    
    dataset_folder_name = os.path.basename(args.train_data_path.rstrip("/"))
    output_dir = f"./lora_outputs/{dataset_folder_name}"
    os.makedirs(output_dir, exist_ok=True)
    
   
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"train_log_{time_str}.log")
    
   
    sys.stdout = LoggerTee(log_file_path)
    sys.stderr = sys.stdout  
    
    print(f"====== Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ======")
    print(f"Dataset identity detected: {dataset_folder_name}")
    print(f"Logs will be saved directly to: {log_file_path}")

    #  加载模型与数据
    model, tokenizer = prepare_model(args.token, args.is_lora, args.lora_rank, args.model_name)
    model.to(device)

    print(f"Loading train dataset from: {args.train_data_path}")
    train_dataset = load_single_split(tokenizer, path=args.train_data_path, split_name="train", max_len=args.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Loading validation dataset from: {args.val_data_path}")
    val_dataset = load_single_split(tokenizer, path=args.val_data_path, split_name="validation", max_len=args.seq_length)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    current_step = 0
    
    print(f"Training model on {device} for {args.epochs} epochs...")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        start_time = time.time()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", file=sys.stdout)):
            loss = train_step(model, optimizer, criterion, batch, device)
            current_step += 1

            if current_step % 100 == 0:
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                total_samples = args.batch_size * 100
                throughput = total_samples / step_time
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
                
                print(f"\nStep: {current_step} | Loss: {loss.item():.4f} | Throughput: {throughput:.2f} smp/s | Mem: {mem:.2f}GB")
                torch.cuda.reset_peak_memory_stats(device)
                
                torch.cuda.synchronize()
                start_time = time.time()

            if current_step % args.eval_every_n_steps == 0:
                val_acc = evaluate(model, val_dataloader, device)
                print(f"\n[Eval] Step {current_step} | Acc: {val_acc * 100:.2f}%")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_model(model, tokenizer, output_dir, "best", time_str)
                
                torch.cuda.synchronize()
                start_time = time.time()

        save_model(model, tokenizer, output_dir, f"epoch_{epoch}", time_str)

    print(f"All done in {time.time() - train_start:.2f}s!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=128)
    
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        default="/home/wrc/sstdp/CusText/CusText/sst2/privatized_dataset/glove_840B-300d/conservative/eps_2.0_top_20_s1_save_stop_words_False",
        help="训练集数据路径"
    )
    parser.add_argument(
        "--val_data_path", 
        type=str, 
        default="/home/wrc/sstdp/CusText/CusText/sst2/datasets/sst2/dev.tsv",
        help="验证集数据路径"
    )
    
    parser.add_argument("--no_lora", dest="is_lora", action="store_false", help="Disable LoRA and train fully")
    parser.set_defaults(is_lora=True)
    
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval_every_n_steps", type=int, default=400)
    parser.add_argument("--gpu_id", type=int, default=0, help="Target GPU index")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/wrc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",  
    )
    parser.add_argument("--token", type=str,default=0,  required=False)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    train(args, device)

if __name__ == "__main__":
    main()