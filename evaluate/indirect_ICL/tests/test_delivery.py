"""CPU checks using synthetic fixtures, independent of prior experiment outputs."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DPICL"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, GPT2Config, GPT2LMHeadModel
from sklearn.metrics import roc_auc_score, roc_curve

from Indirect_attack.neighbour_mia import build_parser, build_prompt, clean_bert_text, load_nonmembers, score_texts
from Indirect_attack.combine import main as combine
from Indirect_attack.evaluate_attack_metrics import compute_metrics
from runtime_config import model_path
from local_llm import Encoding
from task_data import load_wikitext2, text_key


class TinyTokenizer:
    padding_side = "right"

    def __call__(self, texts, max_length=512, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [([1] + [3 + ord(t[0]) % 28 for t in text.split()] + [2])[:max_length]
                for text in texts]
        width = max(map(len, rows))
        left = self.padding_side == "left"
        return BatchEncoding(dict(input_ids=torch.tensor([
            [0] * (width - len(r)) + r if left else r + [0] * (width - len(r)) for r in rows]),
            attention_mask=torch.tensor([
                [0] * (width - len(r)) + [1] * len(r) if left else [1] * len(r) + [0] * (width - len(r))
                for r in rows])))

    def batch_decode(self, ids):
        return [' '.join(map(str, row.tolist())) for row in ids]


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
        for field in ("original_text",):
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
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
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
            for name, target in [("1%", .01), ("0.1%", .001), ("0.01%", .0001)]:
                self.assertEqual(metrics[f"TPR@{name}"], float(max(tpr[fpr <= target])))
                self.assertRegex(output.getvalue(), rf"TPR@{re.escape(name)}FPR: \d+\.\d{{2}}%")
            self.assertFalse({"TPR@100%", "TPR@50%", "TPR@10%"} & set(metrics))

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

    def test_wikitext_snapshot_and_isolation(self):
        data, _ = load_wikitext2(ROOT / "data/WikiText-2-raw-v1")
        self.assertEqual([len(data[split]) for split in ("train", "validation", "test")],
                         [21547, 2208, 2583])
        private = {text_key(text) for text in data["train"]["sentence"]}
        held_out = [text_key(text) for text in data["test"]["sentence"][:500]]
        self.assertEqual(len(held_out), len(set(held_out)))
        self.assertFalse(private.intersection(held_out))

    def test_selected_model_tokenizer_protocol(self):
        selected = os.environ.get("DELIVERY_TEST_MODEL")
        models = [selected] if selected else ["fm9g", "chatglm3"]
        expected = {
            "fm9g": ([1], [73440], [2, 73440]),
            "chatglm3": ([64790, 64792], [2], [2]),
        }
        checked = 0
        for model in models:
            try:
                name = model_path(model)
            except FileNotFoundError:
                if selected:
                    raise
                continue
            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=True,
                                                       local_files_only=True)
            encoding = Encoding(tokenizer)
            if hasattr(tokenizer, "get_prefix_tokens"):
                prefix = tokenizer.get_prefix_tokens()
            else:
                prefix = [tokenizer.bos_token_id] if getattr(tokenizer, "add_bos_token", False) else []
            eos = [tokenizer.eos_token_id] if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id
            self.assertEqual(prefix, expected[model][0])
            self.assertEqual(sorted(eos), expected[model][1])
            config = AutoConfig.from_pretrained(name, trust_remote_code=True, local_files_only=True)
            model_eos = ([config.eos_token_id] if isinstance(config.eos_token_id, int)
                         else config.eos_token_id)
            self.assertEqual(sorted(model_eos), expected[model][2])
            text = "a good film\ncontinued text"
            self.assertEqual(encoding.trim_stops(encoding.encode(text), ["\n"]), "a good film")
            checked += 1
        if not checked:
            self.skipTest("Selected external model is not installed")

    def test_entry_inputs_and_full_dry_run(self):
        env = os.environ.copy()
        env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            for mode, expected in (("attack", "--input-json"), ("evaluate", "--results")):
                result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", mode],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
            for model in ("fm9g", "chatglm3"):
                for dataset in ("sst2", "wikitext2"):
                    output = Path(directory) / f"{model}_{dataset}"
                    result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", "full",
                                             "--model", model, "--dataset", dataset,
                                             "--dry-run", "--output", str(output)],
                                            cwd=directory, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"--target-model {model}", result.stdout)
                    if dataset == "wikitext2":
                        self.assertIn("--task-dataset wikitext2", result.stdout)
                    self.assertIn("--num-valid 10", result.stdout)
                    self.assertIn("--limit 500", result.stdout)
                    self.assertIn("--proposal-model resources/bert-base-uncased", result.stdout)
                    self.assertIn(f"--input-json {output / 'generated/original_text_results.json'}",
                                  result.stdout)
                    self.assertIn(f"--max-length {512 if dataset == 'sst2' else 2048}", result.stdout)
                    self.assertEqual(result.stdout.count('prepare_data:'), int(dataset == 'wikitext2'))
                    self.assertFalse(output.exists())

    def test_entry_evaluation_from_user_results(self):
        env = os.environ.copy()
        env.update(PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for dataset in ("sst2", "wikitext2"):
                for field in ("original_text",):
                    source = p / dataset / "input" / field
                    source.mkdir(parents=True)
                    for name, values in (("other", [.9, .7]), (dataset, [.8, .1])):
                        pd.DataFrame(dict(diff_score=values)).to_csv(
                            source / f"attack_results_{name}_bert_0_{field}.csv", index=False)
                result = subprocess.run(["bash", str(ROOT / "run.sh"), "--mode", "evaluate",
                                         "--dataset", dataset, "--results", str(p / dataset / "input"),
                                         "--output", str(p / dataset / "output")],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for field in ("original_text",):
                    metrics = json.loads((p / dataset / "output" / field / "evaluation/metrics.json").read_text())
                    self.assertAlmostEqual(metrics["AUC"], .75)
                    self.assertEqual(metrics["member_count"], 2)
                    self.assertEqual(metrics["nonmember_count"], 2)


if __name__ == "__main__":
    unittest.main()
