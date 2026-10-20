"""Load and apply saved ChatGLM steering vectors."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path


def validate_steering(steering_dir: str | Path) -> dict:
    directory = Path(steering_dir)
    config_path = directory / "steering_config.json"
    weights_path = directory / "steering.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Steering config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Steering weights not found: {weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "chatglm":
        raise ValueError(f"Expected ChatGLM steering metadata: {config_path}")
    if config.get("method") not in {"mean_steering", "private_steering"}:
        raise ValueError(f"Unknown steering method in {config_path}")
    return config


def load_steering(steering_dir: str | Path):
    from safetensors.torch import load_file

    directory = Path(steering_dir)
    config = validate_steering(directory)
    tensors = load_file(str(directory / "steering.safetensors"), device="cpu")
    directions = {}
    for layer in config["layers"]:
        key = f"layer_{int(layer)}"
        if key not in tensors:
            raise ValueError(f"Missing {key} in steering weights")
        directions[int(layer)] = tensors[key]
    return config, directions


@contextmanager
def apply_steering(model, directions, multiplier: float):
    handles = []
    for layer_index, direction_cpu in directions.items():
        layer = model.transformer.encoder.layers[layer_index]

        def make_hook(direction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                delta = direction.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                updated = hidden + delta * multiplier
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
