from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Sequence

import torch
from tqdm.auto import tqdm


REQUIRED_DATASET_FIELDS = (
    "question",
    "answer_matching_behavior",
    "answer_not_matching_behavior",
)
PRIVATE_NOISE_MULTIPLIER = 0.005


def chat_text(question: str, answer: str = "") -> str:
    """Use the same ChatGLM3 prompt layout for construction and inference."""
    return f"<|user|>\n{question}<|assistant|>\n{answer}"


def contrast_texts(sample: Mapping[str, Any]) -> tuple[str, str]:
    question = str(sample["question"])
    positive = str(sample["answer_matching_behavior"])
    negative = str(sample["answer_not_matching_behavior"])
    return chat_text(question, positive), chat_text(question, negative)


def _layer_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


@dataclass(frozen=True)
class ChatGLMSteeringVector:
    """Layer-indexed steering directions kept on CPU until a request runs."""

    layer_activations: dict[int, torch.Tensor]

    @contextmanager
    def apply(
        self,
        model: torch.nn.Module,
        multiplier: float = 256.0,
        min_token_index: int = 0,
    ) -> Iterator[None]:
        handles: list[Any] = []

        for layer_idx, direction_cpu in self.layer_activations.items():
            layer = model.transformer.encoder.layers[layer_idx]

            def make_hook(direction: torch.Tensor):
                def hook(_module: torch.nn.Module, _inputs: Any, output: Any):
                    hidden = _layer_hidden(output)
                    delta = direction.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ).view(1, 1, -1)
                    delta = delta * multiplier

                    if min_token_index <= 0:
                        updated = hidden + delta
                    else:
                        updated = hidden.clone()
                        updated[min_token_index:] += delta

                    if isinstance(output, tuple):
                        return (updated, *output[1:])
                    return updated

                return hook

            handles.append(layer.register_forward_hook(make_hook(direction_cpu)))

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def _extract_one(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    layers: Sequence[int],
    read_token_index: int,
) -> dict[int, torch.Tensor]:
    device = _model_device(model)
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    for layer_idx in layers:
        layer = model.transformer.encoder.layers[layer_idx]

        def make_hook(idx: int):
            def hook(_module: torch.nn.Module, _inputs: Any, output: Any):
                hidden = _layer_hidden(output)
                if hidden.ndim != 3:
                    raise RuntimeError(
                        f"Expected ChatGLM hidden states with 3 dimensions, got {hidden.shape}."
                    )
                normalized_index = (
                    read_token_index
                    if read_token_index >= 0
                    else hidden.shape[0] + read_token_index
                )
                if normalized_index < 0 or normalized_index >= hidden.shape[0]:
                    raise ValueError(
                        f"Prompt is too short for read_token_index={read_token_index}."
                    )
                captured[idx] = (
                    hidden[read_token_index, 0, :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .clone()
                )

            return hook

        handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    try:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layers).difference(captured)
    if missing:
        raise RuntimeError(f"Failed to capture ChatGLM layers: {sorted(missing)}")
    return captured


def build_mean_and_private_vectors(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    layers: Sequence[int] = (11, 12, 13, 14, 15),
    *,
    read_token_index: int = -2,
    clip: float = 18.0,
    noise_multiplier: float = PRIVATE_NOISE_MULTIPLIER,
    private_seed: int = 42,
    show_progress: bool = True,
) -> tuple[ChatGLMSteeringVector, ChatGLMSteeringVector]:
    """Build mean and private steering from one shared activation pass."""
    if not samples:
        raise ValueError("The steering dataset is empty.")
    if clip <= 0:
        raise ValueError("clip must be greater than zero.")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and non-negative.")
    layer_indices = tuple(dict.fromkeys(int(idx) for idx in layers))
    if not layer_indices:
        raise ValueError("At least one steering layer is required.")

    # Aggregate online so the selected samples do not leave every activation in RAM.
    difference_sums: dict[int, torch.Tensor] = {}
    clipped_sums: dict[int, torch.Tensor] = {}
    max_norms = {idx: 0.0 for idx in layer_indices}

    iterator = tqdm(
        samples,
        desc="Building temporary steering vectors",
        disable=not show_progress,
    )
    for sample in iterator:
        positive_text, negative_text = contrast_texts(sample)
        positive = _extract_one(
            model,
            tokenizer,
            positive_text,
            layer_indices,
            read_token_index,
        )
        negative = _extract_one(
            model,
            tokenizer,
            negative_text,
            layer_indices,
            read_token_index,
        )

        for layer_idx in layer_indices:
            difference = positive[layer_idx] - negative[layer_idx]
            if layer_idx not in difference_sums:
                difference_sums[layer_idx] = torch.zeros_like(difference)
                clipped_sums[layer_idx] = torch.zeros_like(difference)
            difference_sums[layer_idx].add_(difference)

            norm = difference.norm().item()
            max_norms[layer_idx] = max(max_norms[layer_idx], norm)
            clipping_factor = min(1.0, clip / max(norm, 1e-12))
            clipped_sums[layer_idx].add_(difference, alpha=clipping_factor)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(private_seed)
    mean_directions: dict[int, torch.Tensor] = {}
    private_directions: dict[int, torch.Tensor] = {}

    for layer_idx in layer_indices:
        max_norm = max(max_norms[layer_idx], 1e-12)
        average_difference = difference_sums[layer_idx] / len(samples)
        mean_direction = average_difference / max_norm

        clipped_mean = clipped_sums[layer_idx] / (len(samples) * clip)
        noise = torch.randn(
            clipped_mean.shape,
            generator=generator,
            device="cpu",
            dtype=clipped_mean.dtype,
        )
        private_direction = clipped_mean + noise * noise_multiplier

        mean_directions[layer_idx] = mean_direction.detach().cpu()
        private_directions[layer_idx] = private_direction.detach().cpu()

        if show_progress:
            print(
                f"[layer {layer_idx}] "
                f"avg_diff_norm={average_difference.norm().item():.6f}, "
                f"mean_norm={mean_direction.norm().item():.6f}, "
                f"private_signal_norm={clipped_mean.norm().item():.6f}, "
                f"private_noise_norm="
                f"{(noise * noise_multiplier).norm().item():.6f}"
            )

    return (
        ChatGLMSteeringVector(mean_directions),
        ChatGLMSteeringVector(private_directions),
    )
