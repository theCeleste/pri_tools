import json
import shutil
import sys
from pathlib import Path

import torch

LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trainer import OurTrainer  # noqa: E402
from validate_checkpoint import TinyCausalDataset, build_args, build_model  # noqa: E402
from validate_training_matrix import option_collator, wrap_option_loss  # noqa: E402


def privacy_args(output_dir, max_steps=1):
    args = build_args(output_dir, max_steps=max_steps)
    args.trainer = "zo"
    args.zo_dpga = True
    args.zo_dpga_clip_threshold = 1.0
    args.dp_epsilon = 6.0
    args.dp_delta = 1e-5
    args.dataloader_drop_last = True
    return args


def build_privacy_trainer(output_dir, max_steps=1, dataset_size=8):
    model = build_model("lora")
    wrap_option_loss(model, zo_dpga=True)
    return OurTrainer(
        model=model,
        args=privacy_args(output_dir, max_steps=max_steps),
        train_dataset=TinyCausalDataset(size=dataset_size),
        data_collator=option_collator,
    )


def main():
    root = LLAMA_DIR / "tests" / "_checkpoint_validation" / "dp_privacy"
    if root.exists():
        shutil.rmtree(root)

    drop_last_trainer = build_privacy_trainer(root / "drop-last", dataset_size=5)
    if len(drop_last_trainer.get_train_dataloader()) != 2:
        raise AssertionError("ZODPGA dataloader did not drop the final incomplete batch")
    print("PASS ZODPGA fixed-batch drop_last semantics")

    trainer = build_privacy_trainer(root, max_steps=1, dataset_size=8)
    trainer.train()
    checkpoint = root / "checkpoint-1"
    privacy_file = checkpoint / "zo_dpga_privacy.json"
    if not privacy_file.exists():
        raise AssertionError("ZODPGA checkpoint did not save privacy metadata")
    saved = json.loads(privacy_file.read_text(encoding="utf-8"))
    expected = {
        "format": "zo_dpga-privacy-v1",
        "epsilon": 6.0,
        "delta": 1e-5,
        "clip_threshold": 1.0,
        "sample_rate": 0.25,
        "effective_batch_size": 2,
        "num_examples": 8,
        "max_steps": 1,
        "dataloader_drop_last": True,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise AssertionError(f"Privacy manifest field {key}: expected {value}, got {saved.get(key)}")
    if not (saved["noise_multiplier"] > 0 and saved["gaussian_std"] > 0):
        raise AssertionError(f"Invalid DP noise fields: {saved}")
    print("PASS ZODPGA checkpoint privacy manifest")

    exact_resume = build_privacy_trainer(root / "exact", max_steps=1, dataset_size=8)
    exact_resume._load_from_checkpoint(str(checkpoint))
    print("PASS ZODPGA exact-configuration adapter restore")

    changed_resume = build_privacy_trainer(root / "changed", max_steps=2, dataset_size=8)
    try:
        changed_resume._load_from_checkpoint(str(checkpoint))
    except ValueError as error:
        if "privacy-changing arguments" not in str(error):
            raise
    else:
        raise AssertionError("ZODPGA resume accepted a changed max_steps privacy schedule")
    print("PASS ZODPGA rejects privacy-changing resume arguments")

    grouped = {"input_ids": type("ShapeOnly", (), {"shape": (5, 4)})(), "num_options": [2, 2, 3, 3, 3]}
    if OurTrainer._zo_dpga_original_batch_size(grouped) != 2:
        raise AssertionError("Candidate grouping did not recover the original batch cardinality")

    variable_model = build_model("lora")
    wrap_option_loss(variable_model, zo_dpga=True)
    variable_ids = torch.tensor(
        [
            [1, 10, 11, 12, 30],
            [1, 10, 11, 12, 31],
            [1, 20, 21, 22, 40],
            [1, 20, 21, 22, 41],
            [1, 20, 21, 22, 42],
        ],
        dtype=torch.long,
    )
    with torch.inference_mode():
        variable_loss = variable_model(
            input_ids=variable_ids,
            attention_mask=torch.ones_like(variable_ids),
            labels=torch.tensor([1, 1, 2, 2, 2]),
            option_len=[1, 1, 1, 1, 1],
            num_options=[2, 2, 3, 3, 3],
        ).loss
    if variable_loss.shape != (2,) or not torch.isfinite(variable_loss).all():
        raise AssertionError(f"Variable-option DP loss must have shape (2,), got {variable_loss}")
    print("PASS ZODPGA variable-candidate grouping and per-example loss")


if __name__ == "__main__":
    main()
