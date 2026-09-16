
import argparse
import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from huggingface_hub import login  # used to download from HuggingFace
from opacus_patch import apply_bfloat16_patches
apply_bfloat16_patches()
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import LoraConfig, TaskType, get_peft_model  # LORA
from torch.utils.data import DataLoader, DistributedSampler  # , RandomSampler
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from KFprivacy_engine import KF_PrivacyEngine
import sys

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_sst2_dataset(tokenizer, split="train", max_len=128):
    dataset = load_dataset("glue", "sst2", split=split)
    dataset = dataset.filter(lambda example: example["label"] != -1)

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    encoded_dataset = dataset.map(
        tokenize_function, batched=True
    )  # , keep_in_memory=True)
    encoded_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    return encoded_dataset


def prepare_model(
    token: str,
    is_lora: bool = True,
    lora_rank: int = 16,
    model_name: str = "meta-llama/Llama-2-7b-hf",
):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        pretrained_model.config.pad_token_id = tokenizer.eos_token_id
    if is_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Causal language modeling for GPT-style models
            inference_mode=False,  # Enable training mode
            r=lora_rank,  # Low-rank dimension
            lora_alpha=32,  # Alpha scaling factor
            lora_dropout=0.1,  # Dropout for LoRA layers
            target_modules=[
                                "q_proj", 
                                "v_proj", 
                                "k_proj", 
                                "o_proj"
                            ],
        )
        model_with_lora = get_peft_model(pretrained_model, lora_config)

    trainable_parameters = 0
    if is_lora:
        for name, param in model_with_lora.named_parameters():
            if name == ("model.embed_tokens.weight"):
                param.requires_grad = False  # opacus doesn't support positional embedding layer and GC doesn't support tyed parameters
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

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    def closure():
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = criterion(outputs.logits, labels)
        loss.backward()
        return outputs, loss
    outputs, loss = optimizer.step(closure)
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
    model_name: str = "meta-llama/Llama-2-7b-hf",

):
    assert (
        token is not None
    ), "Please provide a valid huggingface token to access gated models"
    
    model, tokenizer = prepare_model(token, is_lora, lora_rank, model_name)
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    # dataset
    train_dataset = prepare_sst2_dataset(tokenizer, split="train", max_len=seq_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(train_dataset),
        num_workers=4,          
        pin_memory=True,        
        prefetch_factor=2
    )
    val_dataset = prepare_sst2_dataset(
        tokenizer, split="validation", max_len=seq_length
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size // world_size,
        sampler=DistributedSampler(val_dataset, shuffle=False),
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    privacy_engine = KF_PrivacyEngine()
    criterion=torch.nn.CrossEntropyLoss()
    model, optimizer,train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_epsilon=8.0,
        epochs=2,
        target_delta=5e-06,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="hooks",
        poisson_sampling=False,
        kalman=True, 
        kappa=0.7, 
        gamma=0.5 
    )
    current_step = 0
    stop_training = False

    best_val_acc = 0.0
    patience =7  
    patience_counter = 0
    stop_training_tensor = torch.tensor(0, device=device) 


    epoch_batch_stats = []
    total_samples_since_last_log = 0
    total_time_since_last_log = 0.0
    
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
                enumerate(memory_safe_data_loader), desc=f"Training epoch {epoch}: ",
                disable=not master_process,file=sys.stdout,total=len(memory_safe_data_loader)
            ):
                
                # ========== 执行训练步骤 ==========
                loss = train_step(model, optimizer, criterion, batch, device)
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
                        # print(
                        #     f"Step: {step} | "
                        #     f"Loss: {loss.item():.4f} | "
                        #     f"Avg Throughput: {throughput:.2f} samples/s | "
                        #     f"Mem: {mem_after:.2f}GB / Peak: {mem_peak:.2f}GB"
                        # )

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
                        with open("./training_metrics.csv", "a") as f:
                            if current_step == eval_every_n_steps: 
                                f.write("step,val_acc,epsilon\n")
                            f.write(f"{current_step},{val_acc},{epsilon},{loss.item()}\n")
                        # 3. 早停逻辑判断
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            patience_counter = 0
                            print(f"New Best Accuracy! Saving best model...")
                            save_private_model(model, tokenizer, "./lora_kal_dp_sst2_model_best", rank)
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

        if master_process:
            epoch_save_path = f"./lora_kal_dp_sst2_model_epoch_{epoch}"
            save_private_model(model, tokenizer, epoch_save_path, rank)
        
        if stop_training_tensor.item() == 1:
            if master_process:
                print(f"Training stopped early at epoch {epoch}")
            break


def launch(
    rank: int,
    world_size: int,
    token: str,
    batch_size: int = 64,
    max_physical_batch_size: int = 64,
    seq_length: int = 128,
    is_lora: bool = True,
    lora_rank: int = 8,
    learning_rate: float = 2e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    model_name: str = "meta-llama/Llama-2-7b-hf",
):

    torch.cuda.set_device(rank)

    setup(rank, world_size)
    master_process = rank == 0 
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
    seed_offset = rank  # each process gets a different seed
    
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

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Distributed Llama Training Arguments")
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
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/wrc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",  
        help="Model name",
    )
    parser.add_argument(
        "--enable_mixed_precision",
        type=bool,
        default=True,
        help="enable mixed precision with bf16",
    )
    parser.add_argument(
        "--token", type=str, default=0, help="Huggingface token", required=False
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(
        f"Is cuda available: {torch.cuda.is_available()}, number of devices: {world_size}"
    )
    if torch.cuda.current_device() == 0:
        print(f"Args: {args}")


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