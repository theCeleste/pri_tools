import math
import shutil
import sys
from functools import partial
from pathlib import Path

import torch


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trainer import OurTrainer  # noqa: E402
from src.utils import forward_wrap_with_option_len_zo_dpga  # noqa: E402
from validate_checkpoint import TinyCausalDataset, build_args, build_model  # noqa: E402


MODES = ("lora", "head", "prefix")
METHODS = ("regular", "mezo", "zo_dpga")


def option_collator(features):
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in ("input_ids", "attention_mask", "labels")
    }
    batch["option_len"] = [3] * len(features)
    return batch


def configure_args(output_dir, method):
    args = build_args(output_dir, max_steps=1)
    args.save_strategy = "no"
    args.evaluation_strategy = "no"
    args.logging_steps = 1
    args.trainer = "regular" if method == "regular" else "zo"
    args.zo_dpga = method == "zo_dpga"
    args.dataloader_drop_last = method == "zo_dpga"
    args.zo_dpga_clip_threshold = 1.0
    args.dp_epsilon = 6.0
    args.dp_delta = 1e-5
    return args


def wrap_option_loss(model, zo_dpga):
    model.original_forward = model.forward
    model.forward = partial(
        forward_wrap_with_option_len_zo_dpga.__get__(model, type(model)),
        zo_dpga=zo_dpga,
    )


def train_one(mode, method, root):
    output_dir = root / f"{mode}-{method}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    model = build_model(mode)
    wrap_option_loss(model, zo_dpga=method == "zo_dpga")
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_before:
        raise AssertionError(f"{mode}/{method}: no trainable parameters")

    trainer = OurTrainer(
        model=model,
        args=configure_args(output_dir, method),
        train_dataset=TinyCausalDataset(size=8),
        data_collator=option_collator,
    )
    result = trainer.train()
    if result.global_step != 1:
        raise AssertionError(f"{mode}/{method}: expected global_step=1, got {result.global_step}")
    if not math.isfinite(result.training_loss):
        raise AssertionError(f"{mode}/{method}: non-finite training loss {result.training_loss}")

    changed = []
    for name, parameter in model.named_parameters():
        if name in trainable_before and not torch.equal(parameter.detach(), trainable_before[name]):
            changed.append(name)
    if not changed:
        raise AssertionError(f"{mode}/{method}: no trainable parameter changed")

    print(
        f"PASS {mode}/{method}: loss={result.training_loss:.6f}, "
        f"trainable={len(trainable_before)} tensors, changed={len(changed)} tensors"
    )


def main():
    root = LLAMA_DIR / "tests" / "_checkpoint_validation" / "training_matrix"
    for mode in MODES:
        for method in METHODS:
            train_one(mode, method, root)


if __name__ == "__main__":
    main()
