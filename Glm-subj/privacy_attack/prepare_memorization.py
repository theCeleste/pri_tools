#!/usr/bin/env python3
"""Create a deterministic Subj-derived canary set for memorization audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from .common import DEFAULT_RAW_DATA, load_split, require_empty_output, write_json, write_jsonl


DEFAULT_COMPLETIONS = {
    "train": "zero zero zero zero zero zero zero zero zero zero zero zero.",
    "validation": "beta beta beta beta beta beta beta beta beta beta beta beta.",
    "test": "gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma.",
}
CONTROL_COMPLETION = (
    "delta delta delta delta delta delta delta delta delta delta delta delta."
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_RAW_DATA))
    parser.add_argument("--output-dir", default="data/subj_memorization_canary")
    parser.add_argument("--records-per-split", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def source_fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["text"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("label", "")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_rows(rows, split, count, seed):
    indexes = list(range(len(rows)))
    random.Random(seed).shuffle(indexes)
    selected = indexes[:count]
    completion = DEFAULT_COMPLETIONS[split]
    generated = []
    for ordinal, source_index in enumerate(selected):
        source = rows[source_index]
        source_hash = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()[:10]
        label_name = "subjective" if int(source.get("label", 0)) else "objective"
        prefix = (
            f"SUBJ audit {split} key {source_hash} item {ordinal:03d} "
            f"label {label_name}. Private phrase:"
        )
        text = f"{prefix} {completion}"
        generated.append({
            "text": text,
            "sentence": text,
            "label": int(source.get("label", 0)),
            "source_text": source["text"],
            "source_split": split,
            "source_index": source_index,
            "canary_id": f"{split}-{ordinal:03d}-{source_hash}",
            "is_memorization_canary": True,
            "audit_prefix": prefix,
            "audit_reference": completion,
            "loss_prefix": prefix,
            "steering_positive": text,
            "steering_negative": f"{prefix} {CONTROL_COMPLETION}",
        })
    return generated, selected


def run(args):
    if args.records_per_split < 1:
        raise ValueError("records-per-split must be positive")
    output = require_empty_output(args.output_dir)
    manifest = {
        "purpose": "controlled_memorization_canary_stress_test",
        "source_data": str(Path(args.data_dir).resolve()),
        "seed": args.seed,
        "records_per_split": args.records_per_split,
        "completions": DEFAULT_COMPLETIONS,
        "control_completion": CONTROL_COMPLETION,
        "scientific_scope": (
            "This synthetic repeated-phrase benchmark verifies that a method can encode and expose "
            "a training-only signal. It is not an estimate of natural Subj leakage."
        ),
        "splits": {},
    }
    for split_index, split in enumerate(("train", "validation", "test")):
        source_rows = load_split(args.data_dir, split)
        if args.records_per_split > len(source_rows):
            raise ValueError(
                f"records-per-split={args.records_per_split} exceeds {split} size {len(source_rows)}"
            )
        rows, source_indexes = make_rows(
            source_rows, split, args.records_per_split, args.seed + split_index,
        )
        write_jsonl(output / f"{split}.jsonl", rows)
        manifest["splits"][split] = {
            "rows": len(rows),
            "source_rows": len(source_rows),
            "source_indexes": source_indexes,
            "source_fingerprint": source_fingerprint(source_rows),
        }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for split in ("train", "validation", "test"):
        load_split(args.data_dir, split)
    return run(args)


if __name__ == "__main__":
    main()
