from __future__ import annotations

import json
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .steering import (
    PRIVATE_NOISE_MULTIPLIER,
    REQUIRED_DATASET_FIELDS,
    ChatGLMSteeringVector,
    build_mean_and_private_vectors,
    chat_text,
)


DEFAULT_LAYERS = (11, 12, 13, 14, 15)
DEFAULT_MULTIPLIER = 256.0
DEFAULT_CONSTRUCTION_SAMPLES = 1000
DEFAULT_NOISE_MULTIPLIER = PRIVATE_NOISE_MULTIPLIER
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "generate_dataset.json"
)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return resolved


def _resolve_dtype(
    dtype: str | torch.dtype | None,
    device: torch.device,
) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float16 if device.type == "cuda" else torch.float32
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def _load_dataset(path: str | Path) -> list[Mapping[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Steering dataset not found: {dataset_path}\n"
            "Copy your dataset to data/generate_dataset.json, or pass dataset_path explicitly."
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or not samples:
        raise ValueError("The steering dataset must be a non-empty JSON list.")

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Dataset row {index} is not a JSON object.")
        missing = [field for field in REQUIRED_DATASET_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"Dataset row {index} is missing fields: {missing}")
        for field in REQUIRED_DATASET_FIELDS:
            if not isinstance(sample[field], str) or not sample[field].strip():
                raise ValueError(
                    f"Dataset row {index}, field {field!r}, must be a non-empty string."
                )
        if "Options:" not in sample["question"]:
            raise ValueError(
                f"Dataset row {index} does not use the expected SUBJ multiple-choice prompt."
            )
        answers = (
            sample["answer_matching_behavior"],
            sample["answer_not_matching_behavior"],
        )
        if not all(answer.startswith(("(A)", "(B)")) for answer in answers):
            raise ValueError(
                f"Dataset row {index} must use full answers beginning with (A) or (B)."
            )
        if answers[0][:3] == answers[1][:3]:
            raise ValueError(f"Dataset row {index} uses the same A/B label twice.")
    return samples


def _move_inputs(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in inputs.items()}


class SteeredChatGLM:
    """A model-like wrapper that applies one in-memory vector per request."""

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        tokenizer: Any,
        steering_vector: ChatGLMSteeringVector,
        lock: threading.RLock,
        multiplier: float = DEFAULT_MULTIPLIER,
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vector = steering_vector
        self.multiplier = float(multiplier)
        self._lock = lock

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @contextmanager
    def applied(self) -> Iterator[torch.nn.Module]:
        """Yield the underlying HF model while this variant's hooks are active."""
        with self._lock, self.steering_vector.apply(
            self.model,
            multiplier=self.multiplier,
        ):
            yield self.model

    def _generation_inputs(
        self,
        prompt: str,
        history: Sequence[Mapping[str, str]] | None,
    ) -> dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "build_chat_input"):
            inputs = self.tokenizer.build_chat_input(
                prompt,
                history=list(history or []),
                role="user",
            )
        else:
            inputs = self.tokenizer(chat_text(prompt), return_tensors="pt")
        return _move_inputs(inputs, self.device)

    def generate(
        self,
        prompt: str,
        *,
        history: Sequence[Mapping[str, str]] | None = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        inputs = self._generation_inputs(prompt, history)
        input_length = inputs["input_ids"].shape[-1]
        options = dict(generation_kwargs)
        options.setdefault("max_new_tokens", max_new_tokens)
        options.setdefault("do_sample", do_sample)

        with torch.inference_mode(), self.applied() as steered_model:
            output = steered_model.generate(**inputs, **options)

        sequences = output.sequences if hasattr(output, "sequences") else output
        new_tokens = sequences[0, input_length:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def _text_log_probability(self, text: str) -> float:
        full_inputs = self.tokenizer(text, return_tensors="pt")
        full_inputs = _move_inputs(full_inputs, self.device)

        output = self.model(**full_inputs, use_cache=False, return_dict=True)
        logits = output.logits.float()
        input_ids = full_inputs["input_ids"]

        if logits.shape[0] != input_ids.shape[0] and logits.shape[1] == input_ids.shape[0]:
            logits = logits.transpose(0, 1)
        if logits.shape[:2] != input_ids.shape:
            raise RuntimeError(
                f"Unexpected logits shape {tuple(logits.shape)} for input {tuple(input_ids.shape)}."
            )

        token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        selected = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        non_special_mask = torch.ones_like(target_ids, dtype=torch.bool)
        for special_id in self.tokenizer.all_special_ids:
            non_special_mask &= target_ids != special_id
        return float(selected[non_special_mask].sum().item())

    def predict(self, sentence: str) -> dict[str, Any]:
        """Return SUBJ A/B probabilities under the selected steering method."""
        question = (
            "Is the following sentence subjective or objective?\n\n"
            f'Sentence: "{sentence}"\n\n'
            "Options:\n"
            "(A) The sentence is subjective.\n"
            "(B) The sentence is objective."
        )
        prefix = chat_text(question)
        answer_a = "(A) The sentence is subjective."
        answer_b = "(B) The sentence is objective."

        with torch.inference_mode(), self.applied():
            # This mirrors the original run_glm.py evaluator. The common prompt
            # contributes the same additive log-probability to A and B.
            score_a = self._text_log_probability(prefix + answer_a)
            score_b = self._text_log_probability(prefix + answer_b)

        probabilities = torch.softmax(torch.tensor([score_a, score_b]), dim=0)
        probability_a = float(probabilities[0].item())
        probability_b = float(probabilities[1].item())
        return {
            "label": "subjective" if probability_a >= probability_b else "objective",
            "subjective_probability": probability_a,
            "objective_probability": probability_b,
            "method": self.name,
            "multiplier": self.multiplier,
        }

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


@dataclass(frozen=True)
class SubjSteeredModels:
    mean: SteeredChatGLM
    private: SteeredChatGLM
    base_model: torch.nn.Module
    tokenizer: Any
    construction_samples: int


def create_subj_models(
    model_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    layers: Sequence[int] = DEFAULT_LAYERS,
    read_token_index: int = -2,
    clip: float = 18.0,
    noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
    multiplier: float = DEFAULT_MULTIPLIER,
    private_seed: int = 42,
    construction_samples: int = DEFAULT_CONSTRUCTION_SAMPLES,
    construction_seed: int = 42,
    show_progress: bool = True,
) -> SubjSteeredModels:
    """
    Load one ChatGLM3 model and build mean/private SUBJ steering in memory.

    Only model_path is required when data/generate_dataset.json is present.
    """
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    resolved_dataset = dataset_path or DEFAULT_DATASET_PATH
    available_samples = _load_dataset(resolved_dataset)
    if construction_samples <= 0:
        raise ValueError("construction_samples must be greater than zero.")
    if len(available_samples) < construction_samples:
        raise ValueError(
            f"The construction dataset needs at least {construction_samples} rows, "
            f"but only {len(available_samples)} were found."
        )
    random.Random(construction_seed).shuffle(available_samples)
    samples = available_samples[:construction_samples]

    if show_progress:
        print(f"Loading ChatGLM model from: {Path(model_path).expanduser()}")
        print(
            f"Construction samples: {len(samples)} "
            f"(selected from {len(available_samples)}, seed={construction_seed})"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if hasattr(tokenizer, "encode_special_tokens"):
        tokenizer.encode_special_tokens = True

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)
    model.eval()

    model_type = getattr(model.config, "model_type", None)
    if model_type != "chatglm":
        raise ValueError(
            f"Expected a ChatGLM checkpoint, but config.model_type={model_type!r}."
        )

    num_layers = len(model.transformer.encoder.layers)
    invalid_layers = [idx for idx in layers if idx < 0 or idx >= num_layers]
    if invalid_layers:
        raise ValueError(
            f"Invalid layers {invalid_layers}; this model has {num_layers} encoder layers."
        )

    mean_vector, private_vector = build_mean_and_private_vectors(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        layers=layers,
        read_token_index=read_token_index,
        clip=clip,
        noise_multiplier=noise_multiplier,
        private_seed=private_seed,
        show_progress=show_progress,
    )

    shared_lock = threading.RLock()
    mean_model = SteeredChatGLM(
        name="mean_steering",
        model=model,
        tokenizer=tokenizer,
        steering_vector=mean_vector,
        lock=shared_lock,
        multiplier=multiplier,
    )
    private_model = SteeredChatGLM(
        name="private_steering",
        model=model,
        tokenizer=tokenizer,
        steering_vector=private_vector,
        lock=shared_lock,
        multiplier=multiplier,
    )
    return SubjSteeredModels(
        mean=mean_model,
        private=private_model,
        base_model=model,
        tokenizer=tokenizer,
        construction_samples=len(samples),
    )
