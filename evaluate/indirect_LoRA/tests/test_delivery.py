"""CPU tests for all four fixed experiment settings."""

import json
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from datasets import load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


class DeliveryTests(unittest.TestCase):
    def test_all_four_commands(self):
        for model in run.MODELS:
            for dataset, (column, template) in run.DATASETS.items():
                with self.subTest(model=model, dataset=dataset):
                    command = run.attack_command(model, dataset, "train", ROOT / "runs/test")
                    self.assertEqual(command[command.index("--dataset-dir") + 1],
                                     str(ROOT / "data" / dataset))
                    self.assertEqual(command[command.index("--target-tokenizer-path") + 1],
                                     str(ROOT / "weights" / model / dataset))
                    self.assertEqual(command[command.index("--text-column") + 1], column)
                    self.assertEqual(command[command.index("--text-template") + 1], template)
                    for flag, value in (("--neighbor-top-k", "8"), ("--max-neighbors", "100"),
                                        ("--score-batch-size", "8"), ("--max-length", "256"),
                                        ("--num-samples", "1200")):
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertIn("--no-resume", command)

    def test_dataset_is_disjoint_from_training(self):
        for dataset, (column, _) in run.DATASETS.items():
            with self.subTest(dataset=dataset):
                train = load_from_disk(str(ROOT / "data" / dataset / "train"))
                other = load_from_disk(str(ROOT / "data" / dataset / "nonmember"))
                members = {normalize(text) for text in train[column]}
                nonmembers = {normalize(text) for text in other[column]}
                self.assertEqual(len(members), len(train))
                self.assertEqual(len(nonmembers), len(other))
                self.assertFalse(members & nonmembers)

    def test_adapter_settings_and_tokenizers(self):
        for model in run.MODELS:
            for dataset in run.DATASETS:
                with self.subTest(model=model, dataset=dataset):
                    folder = ROOT / "weights" / model / dataset
                    config = json.loads((folder / "adapter_config.json").read_text())
                    expected = ({"r": 64, "target_modules": {"q_proj", "k_proj", "v_proj"}}
                                if model == "fm9g" else
                                {"r": 16, "target_modules": {"query_key_value", "dense",
                                                            "dense_h_to_4h", "dense_4h_to_h"}})
                    self.assertEqual(config["r"], expected["r"])
                    self.assertEqual(set(config["target_modules"]), expected["target_modules"])
                    self.assertEqual(config["lora_alpha"], 32)
                    self.assertEqual(config["lora_dropout"], 0)
                    tokenizer = AutoTokenizer.from_pretrained(
                        folder, trust_remote_code=True, local_files_only=True)
                    self.assertIsNotNone(tokenizer.eos_token_id)

    def test_evaluation_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores = root / "scores"
            scores.mkdir()
            for stem, values in (("member", (0.9, 0.7)), ("nonmember", (0.8, 0.1))):
                records = [{"<original_text>: x": values[i % 2], ("neighbour", 0.5): 0.0}
                           for i in range(1200)]
                with (scores / f"{stem}_scores.pkl").open("wb") as stream:
                    pickle.dump(records, stream)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "--model", "fm9g", "--dataset", "sst2",
                 "--mode", "evaluate", "--scores-dir", str(scores),
                 "--output", str(root / "output")], capture_output=True, text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(result.returncode, 0, result.stderr)
            import pandas as pd
            result_csv = pd.read_csv(root / "output/mia_evaluation_results.csv")
            self.assertEqual(len(result_csv), 3)
            self.assertEqual(set(result_csv.target_fpr), {0.01, 0.001, 0.0001})
            self.assertTrue((result_csv.auc == 0.75).all())
            self.assertTrue((result_csv.actual_fpr <= result_csv.target_fpr).all())
            for label in ("1%", "0.1%", "0.01%"):
                self.assertRegex(result.stdout, rf"TPR@{re.escape(label)}FPR: \d+\.\d{{2}}%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
