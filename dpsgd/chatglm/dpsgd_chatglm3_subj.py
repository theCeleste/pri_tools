# ChatGLM3-6B + SUBJ：单卡 LoRA DP-SGD 训练。
# 与 opacus_patch.py 放在同一目录。
# 运行：CUDA_VISIBLE_DEVICES=0 python dpsgd_chatglm3_subj.py --model_name /path/to/chatglm3-6b
# AdamW、LoRA r=16/alpha=32/dropout=0.1、Ghost Clipping、Poisson sampling。
# epsilon=8、delta=5e-6、学习率=1e-4、裁剪阈值=3；当前默认 epochs=4。
# 噪声校准使用实际 epochs；保留原脚本的验证、早停及保存流程。
# train=8000 / validation=1000 / test=1000；训练期间不使用 test。

import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import ClassLabel, load_dataset
from opacus_patch import apply_bfloat16_patches
apply_bfloat16_patches()
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import LoraConfig, get_peft_model  # LORA
from torch.utils.data import DataLoader, DistributedSampler  # , RandomSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from types import MethodType
from peft.tuners.lora import LoraLayer

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_subj_dataset(tokenizer, split="train", max_len=128):
    # 自动下载 SetFit/Subj。train=8000，另将原 test 分层拆成
    # validation=1000、test=1000；早停只使用 validation。
    if split == "train":
        dataset = load_dataset("SetFit/subj", split="train")
    elif split in ("validation", "test"):
        heldout = load_dataset("SetFit/subj", split="test")
        heldout = heldout.cast_column("label", ClassLabel(names=["0", "1"]))
        heldout = heldout.train_test_split(
            test_size=0.5, seed=42, stratify_by_column="label"
        )
        dataset = heldout["train" if split == "validation" else "test"]
    else:
        raise ValueError(f"Unknown split: {split}")

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return encoded_dataset


def _chatglm_lora_forward(self, x, *args, **kwargs):
    # ChatGLM 层间保持 [seq, batch, hidden]，LoRA 内部改用
    # [batch, seq, hidden]，使 Opacus 默认 batch_first=True 正确计算逐样本范数。
    # Linear 在最后一维计算，此转换不改变确定性前向结果。
    result = self._sequence_first_forward(x.transpose(0, 1), *args, **kwargs)
    return result.transpose(0, 1)


def adapt_chatglm_lora_layout(model):
    for module in model.modules():
        if isinstance(module, LoraLayer) and not hasattr(module, "_sequence_first_forward"):
            module._sequence_first_forward = module.forward
            module.forward = MethodType(_chatglm_lora_forward, module)


def prepare_model(
    token: str,
    is_lora: bool = True,
    lora_rank: int = 16,
    model_name: str = "./chatglm3-6b",
):
    pretrained_model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        empty_init=False,
        local_files_only=True,
        output_loading_info=True,
    )
    # 原始对话模型不含分类头；ChatGLM 的 _init_weights 为空，需显式初始化新增头。
    if any("classifier_head" in key for key in loading_info["missing_keys"]):
        pretrained_model.classifier_head.reset_parameters()
    # ChatGLM 的 classifier_head 在源码中可能固定为 FP16，统一到原脚本的 BF16。
    pretrained_model.to(dtype=torch.bfloat16)
    pretrained_model.config.use_cache = False
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    # 官方分类头使用最后一个位置，左侧 padding 保证该位置属于正文。
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pretrained_model.config.pad_token_id = tokenizer.eos_token_id
    if is_lora:
        lora_config = LoraConfig(
            # 通用 PEFT 封装避免额外传入 ChatGLM 不支持的 output_attentions。
            task_type=None,
            inference_mode=False,  # Enable training mode
            r=lora_rank,  # Low-rank dimension
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.1,  # Dropout for LoRA layers
            target_modules=["query_key_value", "dense"],
            modules_to_save=["classifier_head"],
        )
        model_with_lora = get_peft_model(pretrained_model, lora_config)
        adapt_chatglm_lora_layout(model_with_lora)

    trainable_parameters = 0
    if is_lora:
        for name, param in model_with_lora.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  
            if param.requires_grad:
                trainable_parameters += param.numel()

    else:
        for name, param in pretrained_model.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
            if param.requires_grad:
                trainable_parameters += param.numel()

    print(f"Trainable parameters: {trainable_parameters}")
    if is_lora:
        return model_with_lora, tokenizer
    else:
        return pretrained_model, tokenizer


def train_step(model, optimizer, criterion, batch, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss
@torch.no_grad()
def evaluate(model, dataloader, device, rank, world_size):
    model.eval()

    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        preds = torch.argmax(outputs.logits, dim=-1)

        correct += (preds == labels).sum()
        total += labels.numel()

    # DDP 下聚合
    if world_size > 1:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    acc = (correct.float() / total.float()).item()

    model.train()
    return acc

def save_private_model(model, tokenizer, save_path, rank):
    if rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        raw_model = model._module if hasattr(model, "_module") else model
        if isinstance(raw_model, DDP):
            raw_model = raw_model.module
            
        raw_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"LoRA weights saved to {save_path}")

# training loop
def train(
    token: str,
    master_process: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    is_lora: bool = True,
    lora_rank: int = 16,
    seq_length: int = 128,
    batch_size: int = 64,
    max_physical_batch_size: int = 64,
    learning_rate: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    eval_every_n_steps: int = 400,
    model_name: str = "./chatglm3-6b",


):
    # 本地模型和公开 Subj 数据集不需要 Hugging Face token。
    
    model, tokenizer = prepare_model(token, is_lora, lora_rank, model_name)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    # dataset
    train_dataset = prepare_subj_dataset(tokenizer, split="train", max_len=seq_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(train_dataset),
    )
    val_dataset = prepare_subj_dataset(
        tokenizer, split="validation", max_len=seq_length
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(val_dataset, shuffle=False),
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_epsilon=8,
        epochs=epochs,
        target_delta=5e-6,
        max_grad_norm=max_grad_norm,
        criterion=torch.nn.CrossEntropyLoss(),
        grad_sample_mode="ghost",
        poisson_sampling=True
    )
    current_step = 0
    stop_training = False

    best_val_acc = 0.0
    patience = 3 
    patience_counter = 0
    stop_training_tensor = torch.tensor(0, device=device) 


    epoch_batch_stats = []
    total_samples_since_last_log = 0
    total_time_since_last_log = 0.0
    training_seconds = 0.0
    training_samples = 0
    peak_memory_gb = 0.0
    last_checkpoint = None
    
    # 记录初始时间
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    window_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        if world_size > 1:
            dist.broadcast(stop_training_tensor, src=0)
        if stop_training_tensor.item() == 1:
            break
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        with BatchMemoryManager(
            data_loader=train_dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in tqdm(
                enumerate(memory_safe_data_loader), desc=f"Training epoch {epoch}: ", disable=not master_process
            ):
               # ========== 执行训练步骤 ==========
                torch.cuda.synchronize(device)
                step_started = time.perf_counter()
                loss = train_step(model, optimizer, criterion, batch, device)
                torch.cuda.synchronize(device)
                training_seconds += time.perf_counter() - step_started
                training_samples += batch['input_ids'].size(0)
                peak_memory_gb = max(peak_memory_gb, torch.cuda.max_memory_allocated(device) / 1024**3)
                current_step += 1
                

                current_batch_size = batch['input_ids'].size(0)
                total_samples_since_last_log += current_batch_size

                # ========== 记录与输出 ==========
                is_log_step = (step % 50 == 0 and step > 0) 
                
                if is_log_step:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    window_end_time = time.time()
                    
                    elapsed_time = window_end_time - window_start_time
                    
                    throughput = total_samples_since_last_log / elapsed_time
                    
                    mem_after = torch.cuda.memory_allocated(device) / 1024**3 
                    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
                    
                    if master_process:
                        print(
                            f"Step: {step} | "
                            f"Loss: {loss.item():.4f} | "
                            f"Avg Throughput: {throughput:.2f} samples/s | "
                            f"Mem: {mem_after:.2f}GB / Peak: {mem_peak:.2f}GB"
                        )

                        torch.cuda.reset_peak_memory_stats(device)
                    
                    total_samples_since_last_log = 0
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    window_start_time = time.time()

                # 2. 执行评估
                if current_step % eval_every_n_steps == 0:
                    val_acc = evaluate(model, val_dataloader, device, rank, world_size)
                    epsilon = privacy_engine.get_epsilon(delta=5e-6)

                    if master_process:
                        print(f"[Validation] Step {current_step} | Accuracy: {val_acc * 100:.2f}% | Epsilon: {epsilon:.2f}")
                        with open("./dp_chatglm3_subj_metrics.csv", "a") as f:
                            if current_step == eval_every_n_steps: 
                                f.write("step,val_acc,epsilon,loss\n")
                            f.write(f"{current_step},{val_acc},{epsilon},{loss.item()}\n")
                        # 3. 早停逻辑判断
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            patience_counter = 0
                            print(f"New Best Accuracy! Saving best model...")
                            save_private_model(model, tokenizer, "./lora_dp_chatglm3_subj_model_best", rank)
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                        
                        if patience_counter >= patience:
                            print("Early stopping criteria met. Signaling all processes to stop.")
                            stop_training_tensor.fill_(1)


                    if world_size > 1:
                        dist.broadcast(stop_training_tensor, src=0)
                    

                    if stop_training_tensor.item() == 1:
                        break

        # 保存每个 Epoch 的常规 Checkpoint
        if master_process:
            epoch_save_path = f"./lora_dp_chatglm3_subj_model_epoch_{epoch}"
            save_private_model(model, tokenizer, epoch_save_path, rank)
            last_checkpoint = str(Path(epoch_save_path).resolve())
        
        if stop_training_tensor.item() == 1:
            if master_process:
                print(f"Training stopped early at epoch {epoch}")
            break

    if master_process:
        report = {
            "method": "dpsgd", "model": model_name, "dataset": "SetFit/subj",
            "task": "sequence_classification", "seed": 1337,
            "train_samples": len(train_dataset), "validation_samples": len(val_dataset),
            "target_epsilon": 8, "delta": 5e-6,
            "epsilon": privacy_engine.get_epsilon(delta=5e-6),
            "noise_multiplier": optimizer.noise_multiplier,
            "epochs_requested": epochs, "epochs_completed": epoch,
            "physical_steps": current_step, "batch_size": batch_size,
            "max_physical_batch_size": max_physical_batch_size,
            "seq_length": seq_length, "lora_rank": lora_rank,
            "learning_rate": learning_rate, "max_grad_norm": max_grad_norm,
            "training_step_seconds": training_seconds,
            "training_samples_processed": training_samples,
            "training_step_samples_per_second": training_samples / training_seconds if training_seconds else None,
            "peak_memory_allocated_gb": peak_memory_gb,
            "best_validation_accuracy": best_val_acc if current_step >= eval_every_n_steps else None,
            "checkpoint": last_checkpoint,
        }
        Path("training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def launch(
    rank: int,
    world_size: int,
    token: str,
    batch_size: int = 32,
    max_physical_batch_size: int = 16,
    seq_length: int = 128,
    is_lora: bool = True,
    lora_rank: int = 8,
    learning_rate: float = 2e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    model_name: str = "./chatglm3-6b",
):
    # set the device for the current process
    torch.cuda.set_device(rank)
    # setup environment for distributed training
    setup(rank, world_size)
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    def setup_logging(save_path):
        logging.basicConfig(
            filename=os.path.join(save_path, "training.log"),
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    dist.barrier()
    seed_offset = rank  
    
    tokens_per_iter = batch_size * seq_length
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs("/tmp/out", exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    train(
        token,
        master_process,
        rank,
        world_size,
        device=torch.device(f"cuda:{rank}"),
        seq_length=seq_length,
        batch_size=batch_size,
        max_physical_batch_size=max_physical_batch_size,
        lora_rank=lora_rank,
        is_lora=is_lora,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
        model_name=model_name,
    )
    # cleanup process group
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="ChatGLM3-6B x Subj DP-SGD Training Arguments")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per process"
    )
    parser.add_argument(
        "--max_physical_batch_size", type=int, default=16, help="Max physical batch size"
    )
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--is_lora", type=bool, default=True, help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=3.0, help="Max grad norm for DP"
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--output_dir", default=".", help="Directory for checkpoints, CSV and training report")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to a local ChatGLM3-6B model directory",
    )
    
    parser.add_argument(
        "--token", type=str, default=1, help="Huggingface token", required=False
    )
    args = parser.parse_args()
    for name in ("batch_size", "max_physical_batch_size", "seq_length", "lora_rank",
                 "learning_rate", "max_grad_norm", "epochs"):
        if not 0 < getattr(args, name) < float('inf'):
            parser.error(f"--{name} must be positive and finite")
    args.model_name = str(Path(args.model_name).expanduser().resolve())
    world_size = torch.cuda.device_count()
    print(
        f"Is cuda available: {torch.cuda.is_available()}, number of devices: {world_size}"
    )
    if world_size != 1:
        raise RuntimeError(
            "This minimal adaptation supports one GPU. "
            "Run with CUDA_VISIBLE_DEVICES=0 (or another single GPU index)."
        )
    if not args.is_lora:
        raise ValueError("This minimal adaptation requires LoRA; full-parameter DP is not adapted.")
    if torch.cuda.current_device() == 0:
        print(f"Args: {args}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    

    args = (
        world_size,
        args.token,
        args.batch_size,
        args.max_physical_batch_size,
        args.seq_length,
        args.is_lora,
        args.lora_rank,
        args.learning_rate,
        args.max_grad_norm,
        args.epochs,
        args.model_name,
    )
    mp.spawn(
        launch,
        args=args,
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
