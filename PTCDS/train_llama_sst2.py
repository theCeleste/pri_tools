import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from transformers import LlamaForSequenceClassification
from transformers.trainer_utils import IntervalStrategy

from peft import LoraConfig, get_peft_model, TaskType

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "LLaMA 模型路径或 HF 名称，例如 meta-llama/Llama-2-7b-hf 或本地目录"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "HF 缓存目录（可选）"},
    )


@dataclass
class DataArguments:
    train_file: str = field(
        metadata={"help": "PTCDS 处理后的 train.tsv 路径"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "dev/valid.tsv 路径（可选，没有就只训练不验证）"},
    )
    text_column_name: str = field(
        default="sentence",
        metadata={"help": "TSV 里文本所在的列名，SST-2 默认是 sentence"},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "TSV 里标签列名，SST-2 默认是 label"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "最大序列长度"},
    )


def build_label_mapping(dataset, label_column_name):
    labels = dataset[label_column_name]
    unique_labels = sorted(set(str(l) for l in labels))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ====== 每 1000 step eval ======
    if hasattr(training_args, "evaluation_strategy"):
        try:
            training_args.evaluation_strategy = IntervalStrategy.STEPS
        except Exception:
            training_args.evaluation_strategy = "steps"
    if hasattr(training_args, "eval_strategy"):
        try:
            training_args.eval_strategy = IntervalStrategy.STEPS
        except Exception:
            training_args.eval_strategy = "steps"
    if hasattr(training_args, "evaluate_during_training"):
        training_args.evaluate_during_training = True
    if hasattr(training_args, "eval_steps") and training_args.eval_steps is None:
        training_args.eval_steps = 50

    print(
        f"[DEBUG] eval_strategy={getattr(training_args, 'eval_strategy', None)}, "
        f"evaluation_strategy={getattr(training_args, 'evaluation_strategy', None)}, "
        f"eval_steps={getattr(training_args, 'eval_steps', None)}"
    )

    # ====== wandb 集成 ======
    if hasattr(training_args, "report_to"):
        # 你想关掉 wandb 可以自己改这里
        training_args.report_to = ["wandb"]

    if wandb is not None:
        os.environ.setdefault("WANDB_PROJECT", "ptcds-llama2-sst2")
        if getattr(training_args, "run_name", None) is None:
            default_run_name = os.path.basename(
                training_args.output_dir.rstrip("/")) or "llama2-sst2-sanitext"
            training_args.run_name = default_run_name

    # 日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # ================== 读 TSV ==================
    data_files = {"train": data_args.train_file}
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("validation", None)

    # ================== tokenizer ==================
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        cache_dir=model_args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 标签映射
    label2id, id2label = build_label_mapping(train_dataset, data_args.label_column_name)
    num_labels = len(label2id)
    logger.info(f"Label2id: {label2id}")

    # ================== FlashAttention2 + dtype 处理 ==================
    use_flash = True  # 你现在就是想用 FA2

    # flash-attn 只支持 fp16/bf16，所以如果你没开，就自动开 bf16
    if use_flash and not (training_args.bf16 or training_args.fp16):
        logger.warning(
            "You enabled FlashAttention2 but fp32 training is configured. "
            "Setting training_args.bf16 = True so FlashAttention2 can run."
        )
        training_args.bf16 = True

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # ================== 加载 LLaMA 模型 ==================
    if use_flash and (training_args.bf16 or training_args.fp16):
        # 真正开启 FA2 的路径
        model = LlamaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded LLaMA with FlashAttention2, dtype={torch_dtype}.")
    else:
        # 普通 attention（例如你以后想禁用 flash 时可以把 use_flash=False）
        model = LlamaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype if torch_dtype != torch.float32 else None,
        )
        logger.info(f"Loaded LLaMA with standard attention, dtype={torch_dtype}.")

    # 再保险：如果 config 里有这个字段，写一下
    if hasattr(model.config, "attn_implementation") and use_flash and (training_args.bf16 or training_args.fp16):
        model.config.attn_implementation = "flash_attention_2"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.use_cache = False  # 配合 gradient checkpointing

    # ================== LoRA ==================
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ================== tokenize ==================
    def preprocess_function(examples):
        texts = examples[data_args.text_column_name]
        result = tokenizer(
            texts,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
        )
        labels = []
        for l in examples[data_args.label_column_name]:
            l_str = str(l)
            labels.append(label2id[l_str])
        result["labels"] = labels
        return result

    with training_args.main_process_first(desc="tokenize datasets"):
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        if eval_dataset is not None:
            tokenized_eval = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
        else:
            tokenized_eval = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train if training_args.do_train else None,
        eval_dataset=tokenized_eval if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval and tokenized_eval is not None:
        logger.info("*** Final Evaluate ***")
        metrics = trainer.evaluate()
        logger.info(metrics)


if __name__ == "__main__":
    main()
