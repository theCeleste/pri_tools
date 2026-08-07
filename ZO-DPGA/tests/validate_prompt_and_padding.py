import argparse
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import _option_token_length, encode_prompt, forward_wrap_with_option_len_zo_dpga  # noqa: E402
from validate_checkpoint import build_model  # noqa: E402


class ValidationTemplate:
    def encode(self, sample):
        return f"Passage: {sample.data['text']}\nAnswer:"

    def verbalize(self, sample, candidate):
        return f"{self.encode(sample)} {candidate}"

    def encode_sfc(self, sample):
        return "Answer:"

    def verbalize_sfc(self, sample, candidate):
        return f"Answer: {candidate}"


class ValidationTask:
    train_sep = "\n\n"


def sample(text, candidates=("terrible", "great"), correct="great"):
    return SimpleNamespace(
        data={"text": text},
        candidates=list(candidates),
        correct_candidate=correct,
    )


def wrap_for_dp_loss(model):
    model.original_forward = model.forward
    model.forward = partial(
        forward_wrap_with_option_len_zo_dpga.__get__(model, type(model)),
        zo_dpga=True,
    )
    model.eval()
    return model


def assert_token_boundaries(tokenizer):
    pairs = [
        ("Answer:", "Answer: great"),
        ("The sentiment was", "The sentiment was terrible"),
        ("问题：", "问题：正确"),
        ("A", "A multi-token completion"),
    ]
    for prompt, completed in pairs:
        option_len = _option_token_length(tokenizer, prompt, completed)
        full_ids = tokenizer.encode(completed, add_special_tokens=True)
        if not 0 < option_len < len(full_ids):
            raise AssertionError(
                f"Invalid option length for {prompt!r} -> {completed!r}: {option_len}"
            )
    print("PASS Llama tokenizer answer boundaries")


def assert_left_truncation(tokenizer):
    task = ValidationTask()
    template = ValidationTemplate()
    long_sample = sample("context " * 200)
    encodings, option_lens = encode_prompt(
        task,
        template,
        [],
        long_sample,
        tokenizer,
        max_length=32,
    )
    if any(len(ids) > 32 for ids in encodings):
        raise AssertionError("Left truncation exceeded max_length")
    if tokenizer.add_bos_token and any(ids[0] != tokenizer.bos_token_id for ids in encodings):
        raise AssertionError("Left truncation did not preserve the Llama BOS token")
    if any(not 0 < option_len < len(ids) for ids, option_len in zip(encodings, option_lens)):
        raise AssertionError(f"Invalid retained answer spans: {option_lens}")
    print("PASS left truncation preserves BOS and answer spans")


def assert_padding_invariance_and_token_zero():
    model = wrap_for_dp_loss(build_model("lora"))
    short = torch.tensor([[11, 12, 13, 14, 0, 15]], dtype=torch.long)
    short_mask = torch.ones_like(short)
    with torch.inference_mode():
        single_loss = model(
            input_ids=short,
            attention_mask=short_mask,
            labels=short,
            option_len=[3],
        ).loss

    padded_short = torch.tensor([[0, 0, 11, 12, 13, 14, 0, 15]], dtype=torch.long)
    padded_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    other = torch.tensor([[21, 22, 23, 24, 25, 26, 27, 28]], dtype=torch.long)
    batch_ids = torch.cat((padded_short, other), dim=0)
    batch_mask = torch.cat((padded_mask, torch.ones_like(other)), dim=0)
    with torch.inference_mode():
        batch_loss = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            labels=batch_ids,
            option_len=[3, 3],
        ).loss

    torch.testing.assert_close(batch_loss[0], single_loss[0], rtol=1e-5, atol=1e-6)
    if not torch.isfinite(single_loss).all():
        raise AssertionError("A genuine token id 0 in the answer was incorrectly masked")
    print("PASS left-padding invariance and genuine token-id-0 handling")


def assert_candidate_expansion():
    model = wrap_for_dp_loss(build_model("lora"))
    input_ids = torch.tensor(
        [
            [1, 10, 11, 12, 30],
            [1, 10, 11, 12, 31],
            [1, 20, 21, 22, 40],
            [1, 20, 21, 22, 41],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            option_len=[1, 1, 1, 1],
            num_options=[2, 2, 2, 2],
        )
    if outputs.loss.shape != (2,):
        raise AssertionError(
            f"Candidate expansion should return two original-example losses, got {tuple(outputs.loss.shape)}"
        )
    if not torch.isfinite(outputs.loss).all():
        raise AssertionError(f"Candidate classification loss is non-finite: {outputs.loss}")
    print("PASS candidate expansion returns one loss per original example")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if getattr(tokenizer, "_pad_token", None) is None:
        tokenizer.pad_token = tokenizer._unk_token

    assert_token_boundaries(tokenizer)
    assert_left_truncation(tokenizer)
    assert_padding_invariance_and_token_zero()
    assert_candidate_expansion()


if __name__ == "__main__":
    main()
