import sys
from pathlib import Path

import torch


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prefix import PrefixTuning  # noqa: E402
from src.trainer import OurTrainer  # noqa: E402
from validate_checkpoint import TinyCausalDataset, build_args, build_model, collate  # noqa: E402


class MinimalTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [2] if text == "\n" else [3]

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        values = [value for value in values if not skip_special_tokens or value not in (0, 1, 2)]
        return " ".join(str(value) for value in values)


def assert_plain_generation():
    model = build_model("lora")
    model.eval()
    model.config.eos_token_id = None
    model.generation_config.eos_token_id = None
    input_ids = torch.tensor([[1, 7, 8, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    if outputs.shape != (1, 7):
        raise AssertionError(f"Expected three generated tokens, got shape {tuple(outputs.shape)}")
    torch.testing.assert_close(outputs[:, :4], input_ids)
    print("PASS plain autoregressive generation")


def prefix_model(num_prefix=3):
    # Prefix and LoRA are mutually exclusive in run.py; use a fresh base model.
    model = build_model("head")
    PrefixTuning(model, num_prefix=num_prefix, reparam=False, init_by_real_act=True)
    model.eval()
    model.config.eos_token_id = None
    model.generation_config.eos_token_id = None
    return model


def assert_prefix_cache():
    num_prefix = 3
    model = prefix_model(num_prefix)
    input_ids = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    if len(outputs.past_key_values) != model.config.num_hidden_layers:
        raise AssertionError("Prefix cache does not contain one entry per Llama layer")
    expected_length = num_prefix + input_ids.shape[-1]
    lengths = [cache[0].shape[-2] for cache in outputs.past_key_values]
    if any(length != expected_length for length in lengths):
        raise AssertionError(f"Unexpected prefixed cache lengths: {lengths}")
    print("PASS Prefix initial KV cache length")


def assert_prefix_generation_with_left_padding():
    model = prefix_model(num_prefix=3)
    input_ids = torch.tensor(
        [
            [0, 0, 1, 20, 21],
            [1, 30, 31, 32, 33],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    if outputs.shape != (2, 8):
        raise AssertionError(f"Expected a 2x8 generated batch, got {tuple(outputs.shape)}")
    torch.testing.assert_close(outputs[:, :5], input_ids)
    print("PASS Prefix generation with cache continuation and left padding")


def assert_nondiff_generation():
    model = build_model("lora")
    dataset = TinyCausalDataset(size=2, sequence_length=5)
    args = build_args(LLAMA_DIR / "tests" / "_checkpoint_validation" / "generation", max_steps=1)
    args.task_name = "SQuAD"
    args.non_diff = True
    args.max_new_tokens = 2
    args.max_length = 16
    args.sampling = False
    args.temperature = 1.0
    args.num_beams = 1
    args.top_p = 0.95
    args.top_k = None
    args.eos_token = "\n"
    trainer = OurTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
        tokenizer=MinimalTokenizer(),
    )
    batch = collate([dataset[0], dataset[1]])
    batch["gold"] = [["unmatched answer"], ["unmatched answer"]]
    loss = trainer.zo_forward_nondiff(model, batch)
    if loss.shape != () or not torch.isfinite(loss):
        raise AssertionError(f"Expected one finite scalar non-diff loss, got {loss}")
    if not (-1.0 <= loss.item() <= 0.0):
        raise AssertionError(f"Negative F1 objective is outside [-1, 0]: {loss.item()}")
    print("PASS zo_forward_nondiff batched generation objective")


def main():
    torch.manual_seed(42)
    assert_plain_generation()
    assert_prefix_cache()
    assert_prefix_generation_with_left_padding()
    assert_nondiff_generation()


if __name__ == "__main__":
    main()
