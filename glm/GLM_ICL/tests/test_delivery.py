"""CPU checks using synthetic fixtures, independent of prior experiment outputs."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DPICL"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BatchEncoding, GPT2Config, GPT2LMHeadModel
from sklearn.metrics import roc_auc_score, roc_curve

from Indirect_attack.neighbour_mia import build_parser, build_prompt, clean_bert_text, load_nonmembers, score_texts
from Indirect_attack.combine import main as combine
from Indirect_attack.evaluate_attack_metrics import compute_metrics
from runtime_config import model_path
from local_llm import Encoding


class TinyTokenizer:
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [[1] + [3 + ord(t[0]) % 28 for t in text.split()] + [2] for text in texts]
        width = max(map(len, rows))
        left = self.padding_side == "left"
        return BatchEncoding(dict(input_ids=torch.tensor([
            [0] * (width - len(r)) + r if left else r + [0] * (width - len(r)) for r in rows]),
            attention_mask=torch.tensor([
                [0] * (width - len(r)) + [1] * len(r) if left else [1] * len(r) + [0] * (width - len(r))
                for r in rows])))


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.tokenizer = TinyTokenizer()
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=64, n_embd=16,
            n_layer=1, n_head=2)).eval()

    def test_padding_and_batch_consistency(self):
        texts = ["good movie", "a much longer movie review"]
        for side in ("left", "right"):
            self.tokenizer.padding_side = side
            batch = score_texts("Example: ", texts, self.tokenizer, self.model, "cpu")
            single = [score_texts("Example: ", [t], self.tokenizer, self.model, "cpu")[0] for t in texts]
            np.testing.assert_allclose(batch, single, atol=1e-6)

    def test_original_and_neighbor_score_semantics(self):
        args = ("Example: ", ["a good movie"], self.tokenizer, self.model, "cpu")
        self.assertEqual(score_texts(*args, original=True), score_texts(*args))

    def test_native_loss_equivalence(self):
        enc = self.tokenizer("Example: a good movie")
        labels = enc.input_ids.clone()
        labels[:, :2] = -100
        with torch.no_grad():
            expected = -self.model(**enc, labels=labels).loss.item()
        actual = score_texts("Example: ", ["a good movie"], self.tokenizer, self.model, "cpu")[0]
        self.assertAlmostEqual(expected, actual, places=6)

    def test_prompt_and_generated_examples(self):
        data = [dict(original_text=f"original {i}", synthetic_text=f"synthetic {i}",
                     label=["positive"]) for i in range(10)]
        for field in ("original_text", "synthetic_text"):
            expected = "Example:\n" + "".join(
                f"Text:{item[field]}.Label:['positive']\n" for item in data[5:10]
            ) + "\nPlease analyze this text:"
            self.assertEqual(build_prompt(data, field, 5, 10), expected)
        args = build_parser().parse_args(["--dataset", "other"])
        self.assertEqual(args.score_mode, "token-mean")
        self.assertEqual(args.num_neighbors, 10)
        self.assertEqual(args.neighbor_top_k, 6)

    def test_combination_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            pd.DataFrame(dict(diff_score=[.9, .7])).to_csv(p / "member.csv", index=False)
            pd.DataFrame(dict(diff_score=[.8, .1])).to_csv(p / "nonmember.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                combine(["--member-file", str(p / "member.csv"), "--nonmember-file",
                         str(p / "nonmember.csv"), "--output", str(p / "combined.csv")])
                metrics = compute_metrics(p / "combined.csv", plot=False)
            table = pd.read_csv(p / "combined.csv")
            labels, scores = table["label"].values, table["diff_score"].values
            self.assertEqual(metrics["member_count"], 2)
            self.assertEqual(metrics["nonmember_count"], 2)
            self.assertAlmostEqual(metrics["AUC"], .75)
            self.assertAlmostEqual(metrics["AUC"], roc_auc_score(labels, scores))
            fpr, tpr, _ = roc_curve(labels, scores)
            for name, target in [("100%", 1), ("50%", .5), ("10%", .1),
                                 ("1%", .01), ("0.1%", .001), ("0.01%", .0001)]:
                self.assertEqual(metrics[f"TPR@{name}"], float(max(tpr[fpr <= target])))

    def test_sample_sources_and_order(self):
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "resources/bert-base-uncased", use_fast=False,
                                                   local_files_only=True)
        nonmembers = load_nonmembers(str(ROOT / "data/SST-2/test.tsv"))[:500]
        raw = pd.read_csv(ROOT / "data/SST-2/test.tsv", sep="\t")["sentence"].tolist()
        self.assertEqual(nonmembers, [text for text in raw if len(text) > 10][:500])
        self.assertEqual(len(nonmembers), 500)
        encoded = tokenizer(nonmembers, truncation=True, max_length=512)
        decoded = tokenizer.batch_decode(encoded.input_ids)
        normalized = [clean_bert_text(t) for t in decoded]
        self.assertEqual(len(normalized), 500)
        self.assertEqual(normalized, [text.replace(" [SEP]", " ").replace("[CLS] ", " ")
                                      for text in decoded])

    def test_glm_tokenizer_protocol(self):
        try:
            name = model_path("chatglm3")
        except FileNotFoundError:
            self.skipTest("External ChatGLM weights not installed")
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=True,
                                                   local_files_only=True)
        self.assertEqual(tokenizer.get_prefix_tokens(), [64790, 64792])
        encoding = Encoding(tokenizer)
        text = "a good film\ncontinued text"
        self.assertEqual(encoding.trim_stops(encoding.encode(text), ["\n"]), "a good film")
        self.assertEqual(tokenizer.eos_token_id, 2)

    def test_entry_inputs_and_full_dry_run(self):
        env = os.environ.copy()
        env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            for mode, expected in (("attack", "--input-json"), ("evaluate", "--results")):
                result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", mode],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
            result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", "full",
                                     "--dry-run", "--output", str(Path(directory) / "output")],
                                    cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--num-valid 10", result.stdout)
            self.assertIn("--limit 500", result.stdout)
            self.assertIn(f"--input-json {Path(directory) / 'output/generated/synthetic_text_results.json'}",
                          result.stdout)
            self.assertFalse((Path(directory) / "output").exists())

    def test_entry_evaluation_from_user_results(self):
        env = os.environ.copy()
        env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for field in ("original_text", "synthetic_text"):
                source = p / "input" / field
                source.mkdir(parents=True)
                for dataset, values in (("other", [.9, .7]), ("sst2", [.8, .1])):
                    pd.DataFrame(dict(diff_score=values)).to_csv(
                        source / f"attack_results_{dataset}_bert_0_{field}.csv", index=False)
            result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", "evaluate",
                                     "--results", str(p / "input"), "--output", str(p / "output")],
                                    cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for field in ("original_text", "synthetic_text"):
                metrics = json.loads((p / "output" / field / "evaluation/metrics.json").read_text())
                self.assertAlmostEqual(metrics["AUC"], .75)
                self.assertEqual(metrics["member_count"], 2)
                self.assertEqual(metrics["nonmember_count"], 2)


if __name__ == "__main__":
    unittest.main()
