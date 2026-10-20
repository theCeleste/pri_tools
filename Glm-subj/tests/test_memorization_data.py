from pathlib import Path
from types import SimpleNamespace
import unittest

from privacy_attack.build_steering import subj_contrast
from privacy_attack.common import load_split
from privacy_attack.prefix_attack import select_targets


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "subj_memorization_canary"


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


class MemorizationDataTest(unittest.TestCase):
    def test_splits_are_source_aligned_and_disjoint_by_canary(self):
        splits = {name: load_split(DATA_DIR, name) for name in ("train", "validation", "test")}
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 64, "validation": 64, "test": 64,
        })
        references = {}
        for split, rows in splits.items():
            references[split] = {row["audit_reference"] for row in rows}
            for row in rows:
                self.assertEqual(row["source_split"], split)
                self.assertEqual(row["text"], f'{row["audit_prefix"]} {row["audit_reference"]}')
                self.assertTrue(row["source_text"])
        self.assertTrue(references["train"].isdisjoint(references["test"]))

    def test_steering_contrast_uses_canary_fields(self):
        row = load_split(DATA_DIR, "train")[0]
        self.assertEqual(subj_contrast(row), (row["steering_positive"], row["steering_negative"]))

    def test_prefix_attack_uses_explicit_boundary(self):
        row = load_split(DATA_DIR, "train")[0]
        args = SimpleNamespace(prefix_tokens=999, min_suffix_tokens=4, max_new_tokens=24,
                               targets_per_split=1)
        selected = select_targets([row], CharacterTokenizer(), args, seed=42)
        self.assertEqual(selected[0]["explicit_prefix"], row["audit_prefix"])
        self.assertEqual(
            selected[0]["reference_ids"],
            [ord(character) for character in row["audit_reference"][:24]],
        )


if __name__ == "__main__":
    unittest.main()
