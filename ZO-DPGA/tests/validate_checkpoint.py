import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run import OurArguments  # noqa: E402
from src.lora import LoRA  # noqa: E402
from src.prefix import PrefixTuning  # noqa: E402
from src.trainer import OurTrainer  # noqa: E402


class TinyCausalDataset(Dataset):
    def __init__(self, size=8, sequence_length=8, vocab_size=128):
        generator = torch.Generator().manual_seed(1234)
        self.input_ids = torch.randint(
            3, vocab_size, (size, sequence_length), generator=generator, dtype=torch.long
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def collate(features):
    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}


def build_model(mode):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    if mode == "lora":
        LoRA(model, r=2, alpha=4, float16=False)
    elif mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_output_embeddings().parameters():
            parameter.requires_grad = True
    elif mode == "prefix":
        PrefixTuning(
            model,
            num_prefix=3,
            reparam=False,
            init_by_real_act=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model


def build_args(output_dir, max_steps):
    return OurArguments(
        output_dir=str(output_dir),
        model_name="tiny-random-llama",
        task_name="SST2",
        trainer="zo",
        zo_eps=1e-3,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        evaluation_strategy="no",
        save_total_limit=2,
        report_to=[],
        disable_tqdm=True,
        no_cuda=True,
        remove_unused_columns=False,
    )


def train(mode, output_dir, max_steps, resume_from_checkpoint=None):
    model = build_model(mode)
    dataset = TinyCausalDataset()
    trainer = OurTrainer(
        model=model,
        args=build_args(output_dir, max_steps),
        train_dataset=dataset,
        data_collator=collate,
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if result.global_step != max_steps:
        raise AssertionError(f"Expected global_step={max_steps}, got {result.global_step}")
    return model


def checkpoint_state(checkpoint):
    state_file = checkpoint / "trainer_state.json"
    if not state_file.exists():
        raise AssertionError(f"Missing {state_file}")
    return json.loads(state_file.read_text(encoding="utf-8"))


def assert_adapter_checkpoint(checkpoint, mode):
    manifest_file = checkpoint / "adapter_manifest.json"
    weights_file = checkpoint / "pytorch_model.bin"
    if not manifest_file.exists() or not weights_file.exists():
        raise AssertionError(f"{mode}: adapter manifest or weights are missing from {checkpoint}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "zo_dpga-llama-adapter-v1" or manifest.get("mode") != mode:
        raise AssertionError(f"{mode}: invalid adapter manifest: {manifest}")
    try:
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_file, map_location="cpu")
    if set(state_dict) != set(manifest.get("parameter_names", [])):
        raise AssertionError(f"{mode}: weight keys do not match the adapter manifest")
    expected_fragment = {"lora": "lora_", "head": "lm_head", "prefix": "prefix_encoder"}[mode]
    if not state_dict or not all(expected_fragment in name for name in state_dict):
        raise AssertionError(f"{mode}: checkpoint contains non-adapter tensors: {list(state_dict)}")
    if weights_file.stat().st_size >= 2 * 1024 * 1024:
        raise AssertionError(f"{mode}: tiny adapter checkpoint is unexpectedly large")


def assert_best_model_adapter_load(checkpoint, mode, output_dir):
    model = build_model(mode)
    trainer = OurTrainer(
        model=model,
        args=build_args(output_dir, max_steps=1),
        train_dataset=TinyCausalDataset(),
        data_collator=collate,
    )
    trainer.state.best_model_checkpoint = str(checkpoint)
    trainer._load_best_model()
    try:
        expected = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    except TypeError:
        expected = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu")
    actual = model.state_dict()
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name].cpu(), tensor)


def validate_mode(mode, root):
    output_dir = root / mode
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    train(mode, output_dir, max_steps=1)
    checkpoint_1 = output_dir / "checkpoint-1"
    if checkpoint_state(checkpoint_1)["global_step"] != 1:
        raise AssertionError(f"{mode}: checkpoint-1 has an invalid global step")
    assert_adapter_checkpoint(checkpoint_1, mode)
    assert_best_model_adapter_load(checkpoint_1, mode, output_dir)

    resumed_model = train(mode, output_dir, max_steps=2, resume_from_checkpoint=str(checkpoint_1))
    checkpoint_2 = output_dir / "checkpoint-2"
    if checkpoint_state(checkpoint_2)["global_step"] != 2:
        raise AssertionError(f"{mode}: checkpoint-2 has an invalid global step")
    assert_adapter_checkpoint(checkpoint_2, mode)

    trainable_names = [name for name, parameter in resumed_model.named_parameters() if parameter.requires_grad]
    expected_fragment = {"lora": "lora_", "head": "lm_head", "prefix": "prefix_encoder"}[mode]
    if not trainable_names or not all(expected_fragment in name for name in trainable_names):
        raise AssertionError(f"{mode}: unexpected trainable parameters: {trainable_names}")

    print(f"PASS {mode}: resumed checkpoint-1 -> checkpoint-2; trainable={len(trainable_names)} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "lora", "head", "prefix"], default="all")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=LLAMA_DIR / "tests" / "_checkpoint_validation",
    )
    args = parser.parse_args()
    modes = ["lora", "head", "prefix"] if args.mode == "all" else [args.mode]
    for mode in modes:
        validate_mode(mode, args.output_dir)


if __name__ == "__main__":
    main()
