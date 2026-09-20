import os
import inspect
import random
import numpy as np
import torch

from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ============================================================
# 随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)

# ============================================================
# 配置
# ============================================================

GPU_IDS = os.environ.get("GPU_IDS", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "./models/chatglm3-6b"
)

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "./data/sst2"
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./output2/chatglm3_sst2_memorize_final"
)

TASK = os.environ.get("TASK", "sst2").lower()

# ============================================================
# 训练参数 —— 针对快速记忆优化
# ============================================================

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2"))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "64"))

# ============================================================
# LoRA —— 降低秩，减少参数量，训练更快
# ============================================================

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

TARGET_MODULES = [
    item.strip()
    for item in os.environ.get(
        "TARGET_MODULES",
        "query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
    ).split(",")
    if item.strip()
]

# ============================================================
# tokenizer
# ============================================================

print("=" * 60)
print("Loading tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

# ChatGLM3 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"EOS token id: {tokenizer.eos_token_id}")
print(f"PAD token id: {tokenizer.pad_token_id}")

# ============================================================
# model
# ============================================================

print("=" * 60)
print("Loading model")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.use_cache = False

# ============================================================
# dataset
# ============================================================

print("=" * 60)
print(f"Loading dataset for task: {TASK}")
print("=" * 60)

dataset = DatasetDict()

# 不加载 test
for split in ["train", "validation"]:

    split_path = os.path.join(DATA_DIR, split)

    if os.path.exists(split_path):

        dataset[split] = load_from_disk(split_path)

        print(f"{split}: {len(dataset[split])}")

# ============================================================
# format
# ============================================================

print("=" * 60)
print("Formatting dataset")
print("=" * 60)

SST2_LABEL_MAP = {
    0: "negative",
    1: "positive",
}


def format_example(example):

    eos = tokenizer.eos_token or ""

    if TASK == "sst2":
        review = example["sentence"].strip()
        label = SST2_LABEL_MAP[int(example["label"])]

        text = (
            f"Review: {review}\n"
            f"Sentiment: {label}"
            f"{eos}"
        )
    elif TASK == "agnews":
        article = example["text"].strip()
        label_text = example["label_text"].strip()

        text = (
            f"Article: {article}\n"
            f"Topic: {label_text}"
            f"{eos}"
        )
    elif TASK == "wikitext103":
        text = f"{example['text'].strip()}{eos}"
    else:
        raise ValueError(
            f"Unsupported TASK={TASK!r}; expected sst2, agnews, or wikitext103."
        )

    return {
        "text": text
    }


formatted_dataset = DatasetDict()

for split in dataset.keys():

    formatted_dataset[split] = dataset[split].map(
        format_example,
        remove_columns=dataset[split].column_names,
    )

print(formatted_dataset["train"][0]["text"])

# ============================================================
# tokenize
# ============================================================

print("=" * 60)
print("Tokenizing")
print("=" * 60)


def tokenize_function(examples):

    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True,
    )

    return outputs


tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

train_dataset = tokenized_dataset["train"]
val_dataset = tokenized_dataset["validation"]

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# ============================================================
# collator —— 修复：过滤非 tensor 字段，避免 Trainer 报错
# ============================================================

print("=" * 60)
print("Creating collator")
print("=" * 60)


def custom_collator(features):

    # 过滤掉 datasets 可能注入的额外字段
    features = [
        {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
        for f in features
    ]

    max_len = max(len(f["input_ids"]) for f in features)

    input_ids = []
    attention_mask = []

    for f in features:

        ids = f["input_ids"]
        mask = f["attention_mask"]

        pad_len = max_len - len(ids)

        ids = ids + [tokenizer.pad_token_id] * pad_len
        mask = mask + [0] * pad_len

        input_ids.append(ids)
        attention_mask.append(mask)

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long
    )

    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long
    )

    labels = input_ids.clone()

    # padding 不参与 loss
    labels[attention_mask == 0] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    return batch

# ============================================================
# LoRA
# ============================================================

print("=" * 60)
print("Building LoRA")
print("=" * 60)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(
    base_model,
    lora_config
)

model.print_trainable_parameters()

# ============================================================
# training args —— 核心优化：高学习率、少 epoch、关掉验证
# ============================================================

print("=" * 60)
print("Setting training args")
print("=" * 60)

training_args_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.01,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    optim="adamw_torch",
)

training_args_signature = inspect.signature(TrainingArguments)
if "eval_strategy" in training_args_signature.parameters:
    training_args_kwargs["eval_strategy"] = "no"
else:
    training_args_kwargs["evaluation_strategy"] = "no"

training_args = TrainingArguments(**training_args_kwargs)

# ============================================================
# callback —— 更现实的早停阈值
# ============================================================

from transformers import TrainerCallback


class MemorizationCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None:
            return

        if "loss" in logs:

            loss = logs["loss"]

            # 防止 exp overflow
            ppl = np.exp(min(loss, 20))

            print(
                f"[Step {state.global_step}] "
                f"Train Loss={loss:.4f} "
                f"PPL={ppl:.4f}"
            )

            # 达到记忆目标（对短文本 1.5 已足够证明记住）
            if ppl < 1.5:

                print("=" * 60)
                print("MEMORIZATION TARGET REACHED")
                print("=" * 60)

                control.should_training_stop = True

        if "eval_loss" in logs:

            eval_loss = logs["eval_loss"]

            eval_ppl = np.exp(min(eval_loss, 20))

            print(
                f"[Eval] "
                f"Loss={eval_loss:.4f} "
                f"PPL={eval_ppl:.4f}"
            )

# ============================================================
# trainer
# ============================================================

print("=" * 60)
print("Building trainer")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=custom_collator,
    callbacks=[
        MemorizationCallback()
    ]
)

# ============================================================
# train
# ============================================================

print("=" * 60)
print("Start training")
print("=" * 60)

trainer.train()

# ============================================================
# save
# ============================================================

print("=" * 60)
print("Saving model")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved to: {OUTPUT_DIR}")

# ============================================================
# summary
# ============================================================

print("=" * 60)
print("Training summary")
print("=" * 60)

final_loss = None

for log in trainer.state.log_history:

    if "loss" in log:
        final_loss = log["loss"]

if final_loss is not None:

    final_ppl = np.exp(min(final_loss, 20))

    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final PPL : {final_ppl:.4f}")

    if final_ppl < 1.5:
        print("SUCCESS: model memorized training data")
    else:
        print("Memorization target not reached")
