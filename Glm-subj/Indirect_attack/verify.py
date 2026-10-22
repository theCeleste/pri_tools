from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "run.sh",
    "run.py",
    "evaluate.py",
    "METHOD.md",
    "requirements.txt",
    "example/code/run_steering.py",
    "example/code/select_subset.py",
    "example/interface/data/generate_dataset.json",
    "example/interface/data/test_dataset_ab.json",
    "embedding/data/splits/train.jsonl",
    "embedding/data/splits/eval.jsonl",
    "embedding/weights/raw/adapter_model.safetensors",
    "embedding/weights/cliptext_lora/adapter_model.safetensors",
    "gradient/data/splits/train.jsonl",
    "gradient/data/splits/eval.jsonl",
    "gradient/weights/mezo/adapter_model.safetensors",
    "gradient/weights/dpzero/adapter_model.safetensors",
    "resources/bert-base-uncased/model.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify delivery contents")
    parser.add_argument("--model", default="../models/chatglm3-6b")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    args = parse_args()
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing file: {relative}")

    expected_counts = {
        "embedding/data/splits/train.jsonl": 100,
        "embedding/data/splits/eval.jsonl": 1000,
        "gradient/data/splits/train.jsonl": 100,
        "gradient/data/splits/eval.jsonl": 1000,
    }
    for relative, expected in expected_counts.items():
        path = ROOT / relative
        if path.is_file() and count_jsonl(path) != expected:
            errors.append(f"Unexpected row count: {relative}")

    forbidden_parts = {"__pycache__", ".pytest_cache"}
    forbidden_suffixes = {".log", ".pid", ".pyc"}
    forbidden_names = {"scores.csv", "metrics.json", "metrics.csv"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == "runs":
            continue
        if forbidden_parts & set(relative.parts):
            errors.append(f"Unexpected runtime directory: {relative}")
        if path.is_file() and (path.suffix in forbidden_suffixes or path.name in forbidden_names):
            errors.append(f"Unexpected runtime file: {relative}")

    model = Path(args.model)
    if not model.is_absolute():
        model = (ROOT / model).resolve()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if not (model / name).is_file():
            errors.append(f"Model file missing: {model / name}")

    for module in ("torch", "transformers", "peft", "numpy", "pandas", "sklearn", "sentencepiece"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Cannot import {module}: {exc}")

    checksum_path = ROOT / "checksums.json"
    if not args.skip_checksums:
        if not checksum_path.is_file():
            errors.append("Missing checksums.json")
        else:
            records = json.loads(checksum_path.read_text(encoding="utf-8"))
            for relative, expected in records.items():
                path = ROOT / relative
                if not path.is_file():
                    errors.append(f"Checksum file missing: {relative}")
                elif digest(path) != expected:
                    errors.append(f"Checksum mismatch: {relative}")

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Verification passed.")
    print(f"Base model: {model}")


if __name__ == "__main__":
    main()
