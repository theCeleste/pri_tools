from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWERS = (
    " objective",
    " subjective",
)


def prompt_for(text: str) -> str:
    return (
        f"Sentence: {text.strip()}\n"
        "Question: Is this sentence subjective or objective?\n"
        "Answer:"
    )


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_tokenizer(model_path: str | Path):
    return AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )


def load_base_model(model_path: str | Path, device: str, dtype: str = "bf16"):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=_dtype(dtype),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(torch.device(device))
    return model


def attach_lora(model, *, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query_key_value"],
        bias="none",
    )
    wrapped = get_peft_model(model, config)
    trainable = [(name, parameter) for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable ChatGLM LoRA parameters were created")
    return wrapped


def load_adapter_model(
    model_path: str | Path,
    adapter_path: str | Path,
    device: str,
    dtype: str = "bf16",
):
    base = load_base_model(model_path, device, dtype)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()
    return model


def _batch_first_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] == input_ids.shape:
        return logits
    if logits.ndim == 3 and logits.shape[0] == input_ids.shape[1] and logits.shape[1] == input_ids.shape[0]:
        return logits.transpose(0, 1)
    raise RuntimeError(f"Unexpected logits/input shapes: {tuple(logits.shape)} vs {tuple(input_ids.shape)}")


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def _padded_answer_starts(
    attention: torch.Tensor,
    answer_starts: Sequence[int],
    padding_side: str,
) -> torch.Tensor:
    starts = torch.tensor(answer_starts, device=attention.device)
    if padding_side == "left":
        starts = starts + (attention.shape[1] - attention.sum(1))
    return starts.unsqueeze(1)


@torch.inference_mode()
def classification_losses(model, tokenizer, rows: Sequence[dict], max_length: int) -> torch.Tensor:
    if not rows:
        return torch.empty(0, device=next(model.parameters()).device)
    texts = []
    answer_starts = []
    targets = []
    for row in rows:
        prompt = prompt_for(row["text"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        for answer in ANSWERS:
            full = prompt + answer
            full_ids = tokenizer(full, add_special_tokens=True).input_ids
            answer_starts.append(_common_prefix_length(prompt_ids, full_ids))
            texts.append(full)
        targets.append(int(row["label"]))

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention = encoded.get("attention_mask", torch.ones_like(input_ids))
    outputs = model(**encoded, use_cache=False, return_dict=True)
    logits = _batch_first_logits(outputs.logits.float(), input_ids)
    labels = input_ids[:, 1:]
    token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    selected = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    absolute_positions = torch.arange(1, input_ids.shape[1], device=device).unsqueeze(0)
    starts = _padded_answer_starts(attention, answer_starts, tokenizer.padding_side)
    valid = attention[:, 1:].bool() & attention[:, :-1].bool() & (absolute_positions >= starts)
    for token_id in tokenizer.all_special_ids:
        valid &= labels != token_id
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError("No answer tokens remain after tokenization")
    candidate_scores = (selected * valid).sum(1) / counts
    candidate_scores = candidate_scores.reshape(len(rows), len(ANSWERS))
    target_tensor = torch.tensor(targets, device=device)
    return F.cross_entropy(candidate_scores, target_tensor, reduction="none")


@torch.inference_mode()
def causal_lm_loss_components(
    model,
    tokenizer,
    rows: Sequence[dict],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        device = next(model.parameters()).device
        return torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)
    encoded = tokenizer(
        [str(row["text"]).strip() for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention = encoded.get("attention_mask", torch.ones_like(input_ids))
    logits = _batch_first_logits(
        model(**encoded, use_cache=False, return_dict=True).logits.float(), input_ids
    )
    labels = input_ids[:, 1:]
    valid = attention[:, 1:].bool() & attention[:, :-1].bool()
    for token_id in tokenizer.all_special_ids:
        valid &= labels != token_id
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError("No valid causal-LM target tokens remain after tokenization")
    token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    selected = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    negative_log_likelihoods = -(selected * valid).sum(1)
    return negative_log_likelihoods, counts


def causal_lm_losses(model, tokenizer, rows: Sequence[dict], max_length: int) -> torch.Tensor:
    negative_log_likelihoods, counts = causal_lm_loss_components(
        model, tokenizer, rows, max_length
    )
    return negative_log_likelihoods / counts


@torch.inference_mode()
def predict_labels(model, tokenizer, rows: Sequence[dict], max_length: int) -> list[int]:
    if not rows:
        return []
    texts = []
    starts = []
    for row in rows:
        prompt = prompt_for(row["text"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        for answer in ANSWERS:
            full = prompt + answer
            full_ids = tokenizer(full, add_special_tokens=True).input_ids
            starts.append(_common_prefix_length(prompt_ids, full_ids))
            texts.append(full)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention = encoded.get("attention_mask", torch.ones_like(input_ids))
    logits = _batch_first_logits(model(**encoded, use_cache=False, return_dict=True).logits.float(), input_ids)
    labels = input_ids[:, 1:]
    selected = F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, input_ids.shape[1], device=device).unsqueeze(0)
    padded_starts = _padded_answer_starts(attention, starts, tokenizer.padding_side)
    valid = attention[:, 1:].bool() & attention[:, :-1].bool() & (positions >= padded_starts)
    for token_id in tokenizer.all_special_ids:
        valid &= labels != token_id
    scores = ((selected * valid).sum(1) / valid.sum(1)).reshape(len(rows), 2)
    candidate = scores.argmax(1).cpu().tolist()
    return candidate


@torch.inference_mode()
def sequence_log_probabilities(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    loss_prefix_tokens: int = 0,
) -> list[float]:
    device = next(model.parameters()).device
    results: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention = encoded.get("attention_mask", torch.ones_like(input_ids))
        logits = _batch_first_logits(model(**encoded, use_cache=False, return_dict=True).logits.float(), input_ids)
        labels = input_ids[:, 1:]
        valid = attention[:, 1:].bool() & attention[:, :-1].bool()
        if loss_prefix_tokens:
            special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
            context_targets = special_token_count - 1 + loss_prefix_tokens
            valid &= valid.cumsum(dim=1) > context_targets
        else:
            for token_id in tokenizer.all_special_ids:
                valid &= labels != token_id
        token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        selected = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        counts = valid.sum(1)
        if (counts == 0).any():
            raise ValueError("No valid target tokens remain")
        results.extend(((selected * valid).sum(1) / counts).cpu().tolist())
    return results


def chunks(rows: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]
