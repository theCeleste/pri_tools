from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def normalized_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert(row: dict, source_split: str, source_index: int, membership: str) -> dict:
    return {
        "source_split": source_split,
        "source_index": int(source_index),
        "text": str(row["sentence"]).strip(),
        "label": int(row["label"]),
        "membership": membership,
    }


def prepare_splits(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    train_count: int = 3000,
    dev_count: int = 500,
    eval_count: int = 1000,
    split_seed: int = 0,
) -> dict:
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    raw_train = read_jsonl(source_dir / "train.jsonl")
    raw_test = read_jsonl(source_dir / "test.jsonl")
    if train_count + dev_count > len(raw_train):
        raise ValueError("Requested train/dev rows exceed the SUBJ training pool")
    if eval_count > len(raw_test):
        raise ValueError("Requested evaluation rows exceed the SUBJ test pool")

    train_order = np.random.RandomState(split_seed).permutation(len(raw_train))
    eval_order = np.random.RandomState(split_seed).permutation(len(raw_test))
    member_ids = train_order[:train_count]
    dev_ids = train_order[train_count : train_count + dev_count]
    nonmember_ids = eval_order[:eval_count]

    members = [_convert(raw_train[i], "train", int(i), "member") for i in member_ids]
    dev = [_convert(raw_train[i], "train", int(i), "dev") for i in dev_ids]
    nonmembers = [_convert(raw_test[i], "test", int(i), "nonmember") for i in nonmember_ids]

    groups = {"train": members, "dev": dev, "eval": nonmembers}
    normalized = {
        name: [normalized_text(row["text"]) for row in rows]
        for name, rows in groups.items()
    }
    for name, values in normalized.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate normalized texts in {name} split")
    for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        overlap = set(normalized[left]) & set(normalized[right])
        if overlap:
            raise ValueError(f"{left}/{right} overlap contains {len(overlap)} texts")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in groups.items():
        write_jsonl(rows, output_dir / f"{name}.jsonl")
    manifest = {
        "dataset": "SUBJ",
        "split_seed": split_seed,
        "counts": {name: len(rows) for name, rows in groups.items()},
        "source": {
            "train": str(source_dir / "train.jsonl"),
            "test": str(source_dir / "test.jsonl"),
        },
        "source_sha256": {
            "train": _sha256(source_dir / "train.jsonl"),
            "test": _sha256(source_dir / "test.jsonl"),
        },
        "source_indices": {
            "train": [int(i) for i in member_ids],
            "dev": [int(i) for i in dev_ids],
            "eval": [int(i) for i in nonmember_ids],
        },
        "normalized_overlap": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_subset_splits(
    reference_dir: str | Path,
    output_dir: str | Path,
    *,
    train_count: int,
    dev_count: int,
    eval_count: int,
) -> dict:
    reference_dir = Path(reference_dir).resolve()
    output_dir = Path(output_dir).resolve()
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    reference_train, reference_dev, reference_eval = load_splits(reference_dir)
    if train_count > len(reference_train):
        raise ValueError("Requested train subset exceeds the reference member split")
    if dev_count > len(reference_dev) or eval_count > len(reference_eval):
        raise ValueError("Requested dev/eval subset exceeds the reference split")

    groups = {
        "train": reference_train[:train_count],
        "dev": reference_dev[:dev_count],
        "eval": reference_eval[:eval_count],
    }
    normalized = {
        name: [normalized_text(row["text"]) for row in rows]
        for name, rows in groups.items()
    }
    for name, values in normalized.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate normalized texts in {name} split")
    for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        overlap = set(normalized[left]) & set(normalized[right])
        if overlap:
            raise ValueError(f"{left}/{right} overlap contains {len(overlap)} texts")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in groups.items():
        write_jsonl(rows, output_dir / f"{name}.jsonl")
    manifest = {
        "dataset": "SUBJ",
        "split_seed": reference_manifest["split_seed"],
        "counts": {name: len(rows) for name, rows in groups.items()},
        "reference_splits": str(reference_dir),
        "reference_manifest_sha256": _sha256(reference_dir / "manifest.json"),
        "source_indices": {
            name: [int(row["source_index"]) for row in rows]
            for name, rows in groups.items()
        },
        "normalized_overlap": 0,
        "construction": "train prefix subset; dev and eval preserved from reference",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_splits(split_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    split_dir = Path(split_dir)
    return tuple(
        read_jsonl(split_dir / f"{name}.jsonl") for name in ("train", "dev", "eval")
    )
