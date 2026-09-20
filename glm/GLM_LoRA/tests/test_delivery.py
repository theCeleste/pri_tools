"""CPU tests using fixed input data and synthetic scores, not historical outputs."""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import load_from_disk
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, BatchEncoding, GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from attack import compute_slice, format_target_text, score_texts
from evaluate import analyze_single_text, best_threshold_for_fpr, evaluate, rate_at_threshold

spec = importlib.util.spec_from_file_location("entry", ROOT / "run.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [[1] + [3 + ord(t[0]) % 28 for t in text.split()] + [2] for text in texts]
        width = max(map(len, rows))
        return BatchEncoding(dict(
            input_ids=torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])))


class DeliveryTests(unittest.TestCase):
    def test_external_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CHATGLM_MODEL_DIR": directory}):
                self.assertEqual(entry.base_model_dir(), Path(directory).resolve())
                command = entry.build_attack(ROOT / "weights/lora", "train", ROOT / "runs/test")
                self.assertEqual(command[command.index("--target-model-path") + 1], directory)
            env = os.environ.copy()
            env["CHATGLM_MODEL_DIR"] = str(Path(directory) / "missing")
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--mode", "attack", "--dry-run"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("set CHATGLM_MODEL_DIR", result.stderr)

    def test_dataset_and_nonmember_exclusion(self):
        train = load_from_disk(str(ROOT / "data/train"))
        nonmember = load_from_disk(str(ROOT / "data/nonmember"))
        validation = load_from_disk(str(ROOT / "data/validation"))
        self.assertEqual((len(train), len(nonmember), len(validation)), (33579, 33399, 872))
        keys = [set(normalize(t) for t in split["sentence"]) for split in (train, nonmember)]
        self.assertEqual(len(keys[0]), len(train))
        self.assertEqual(len(keys[1]), len(nonmember))
        self.assertFalse(keys[0] & keys[1])
        self.assertTrue(all(str(t).strip() for split in (train, nonmember)
                            for t in split.select(range(1200))["sentence"]))

    def test_weights_and_tokenizers(self):
        config = json.loads((ROOT / "weights/lora/adapter_config.json").read_text())
        self.assertEqual(config["r"], 16)
        self.assertEqual(config["lora_alpha"], 32)
        self.assertEqual(config["lora_dropout"], 0)
        self.assertEqual(set(config["target_modules"]),
                         {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"})
        with safe_open(ROOT / "weights/lora/adapter_model.safetensors", framework="pt") as f:
            self.assertTrue(f.keys())
            self.assertTrue(all("lora_" in key for key in f.keys()))
        base = entry.base_model_dir()
        index = json.loads((base / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            self.assertTrue((base / name).is_file())
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights/lora", trust_remote_code=True,
                                                   local_files_only=True)
        self.assertEqual(tokenizer.get_prefix_tokens(), [64790, 64792])
        proposal = AutoTokenizer.from_pretrained(ROOT / "resources/roberta-base", local_files_only=True)
        self.assertIsNotNone(proposal.mask_token_id)

    def test_slice_and_attack_format(self):
        args = argparse.Namespace(proc_id=None, start_index=0, end_index=None, num_samples=1200)
        self.assertEqual(compute_slice(33579, args), (0, 1200))
        args = argparse.Namespace(label_column="label", text_template="sentence", add_eos=True)
        tokenizer = argparse.Namespace(eos_token="</s>")
        self.assertEqual(format_target_text(" a good film ", {"label": 1}, args, tokenizer),
                         "a good film</s>")

    def test_training_format_and_collator(self):
        tokenizer = AutoTokenizer.from_pretrained(entry.base_model_dir(), trust_remote_code=True,
                                                   local_files_only=True)
        parsed = ast.parse((ROOT / "code/train.py").read_text(encoding="utf-8"))
        selected = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
                                   and node.name in {"format_example", "custom_collator"}], type_ignores=[])
        namespace = dict(tokenizer=tokenizer, torch=torch, TASK="sst2",
                         SST2_LABEL_MAP={0: "negative", 1: "positive"})
        exec(compile(selected, "train_helpers", "exec"), namespace)
        text = namespace["format_example"]({"sentence": " a good film ", "label": 1})["text"]
        self.assertEqual(text, f"Review: a good film\nSentiment: positive{tokenizer.eos_token}")
        features = [dict(input_ids=[1, 2], attention_mask=[1, 1]),
                    dict(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])]
        batch = namespace["custom_collator"](features)
        self.assertEqual(batch["labels"].tolist(), [[1, 2, -100], [1, 2, 3]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])

    def test_token_mean_loss(self):
        torch.manual_seed(4)
        tokenizer = TinyTokenizer()
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=64, n_embd=16,
                                         n_layer=1, n_head=2)).eval()
        texts = ["good movie", "a much longer movie review"]
        encoded = tokenizer(texts)
        with torch.no_grad():
            logits = model(**encoded).logits[:, :-1].float()
            labels = encoded.input_ids[:, 1:].clone()
            labels[encoded.attention_mask[:, 1:] == 0] = -100
            expected = []
            for row, target in zip(logits, labels):
                expected.append(-torch.nn.functional.cross_entropy(row, target, ignore_index=-100).item())
        actual = score_texts(texts, tokenizer, model, torch.device("cpu"), 256)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_attack_gap_and_strict_threshold(self):
        item = {"<original_text>: example": -1.0, ("neighbor1", .5): -2.0,
                ("neighbor2", .3): -4.0}
        self.assertEqual(analyze_single_text(item)["score_gap"], 2.0)
        members = np.array([.9, .7])
        nonmembers = np.array([.8, .1])
        self.assertEqual(rate_at_threshold(members, nonmembers, .8), (.5, 0., 1, 0, 2, 1))
        result = best_threshold_for_fpr(members, nonmembers, 0)
        self.assertEqual(result[:3], (.8, .5, 0.))

    def test_metrics_from_synthetic_scores(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name, values in (("member", [.9, .7]), ("nonmember", [.8, .1])):
                items = [{"<original_text>: example": value, ("neighbor", .5): 0.}
                         for value in values]
                with (p / f"{name}.pkl").open("wb") as stream:
                    pickle.dump(items, stream)
            with contextlib.redirect_stdout(io.StringIO()):
                frame = evaluate(str(p / "member.pkl"), str(p / "nonmember.pkl"),
                                 [1., .5, .1, .01, .001, .0001], str(p / "metrics.csv"))
            self.assertEqual(len(frame), 6)
            self.assertAlmostEqual(frame.iloc[0]["auc"], roc_auc_score([1, 1, 0, 0], [.9, .7, .8, .1]))
            self.assertTrue((frame.actual_fpr <= frame.target_fpr).all())

    def test_entry_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--mode", "full",
                                     "--gpu", "2", "--output", str(Path(directory) / "output"),
                                     "--dry-run"], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("physical_gpu=2", result.stdout)
            self.assertIn("--num-samples 1200", result.stdout)
            self.assertIn("--max-neighbors 100", result.stdout)
            self.assertFalse((Path(directory) / "output").exists())
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--mode", "evaluate"],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--scores-dir", result.stderr)
        command = entry.build_attack(ROOT / "weights/lora", "train", ROOT / "runs/test")
        self.assertIn("--target-tokenizer-path", command)
        self.assertIn("--no-resume", command)

    def test_evaluate_entry_and_validation(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            source = p / "scores"
            source.mkdir()
            for name, values in (("member", [.9, .7]), ("nonmember", [.8, .1])):
                records = [{"<original_text>: example": values[index % 2], ("neighbor", .5): 0.}
                           for index in range(1200)]
                with (source / f"{name}_scores.pkl").open("wb") as stream:
                    pickle.dump(records, stream)
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--mode", "evaluate",
                                     "--scores-dir", str(source), "--output", str(p / "output")],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            import pandas as pd
            frame = pd.read_csv(p / "output/mia_evaluation_results.csv")
            self.assertTrue((frame.member_count == 1200).all())
            self.assertTrue((frame.nonmember_count == 1200).all())
            self.assertTrue((frame.auc == .75).all())
            records[0]["<original_text>: example"] = float("nan")
            with (source / "nonmember_scores.pkl").open("wb") as stream:
                pickle.dump(records, stream)
            with self.assertRaisesRegex(ValueError, "Invalid score"):
                entry.validate_scores(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
