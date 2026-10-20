from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = Path("/home/wym/projects/dpfinetune/chatglm3-6b")
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "subj_raw_prepared" / "subj" / "none_a3c561096c597e10"
DEFAULT_EMBEDDING_DATA = PROJECT_ROOT / "data" / "subj_embedding_protected"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            text = value.get("text", value.get("sentence"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no non-empty text/sentence")
            label = value.get("label")
            if label is not None and int(label) not in (0, 1):
                raise ValueError(f"{path}:{line_number} has invalid Subj label {label!r}")
            rows.append({**value, "text": text.strip()})
    if not rows:
        raise ValueError(f"Dataset split is empty: {path}")
    return rows


def load_split(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    for suffix in (".jsonl", ".json"):
        path = data_dir / f"{split}{suffix}"
        if not path.is_file():
            continue
        if suffix == ".jsonl":
            return read_jsonl(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list: {path}")
        rows = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item {index} is not an object")
            text = item.get("text", item.get("sentence"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: item {index} has no text/sentence")
            rows.append({**item, "text": text.strip()})
        return rows
    raise FileNotFoundError(f"No {split}.jsonl or {split}.json in {data_dir}")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def require_empty_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_chatglm_base(base_model: str | Path) -> dict[str, Any]:
    path = Path(base_model) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Base model config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("model_type") != "chatglm":
        raise ValueError(f"Expected ChatGLM, found {config.get('model_type')!r}")
    return config


def validate_adapter(adapter_dir: str | Path, expected_task: str = "CAUSAL_LM") -> dict[str, Any]:
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    actual = str(config.get("task_type", "")).upper()
    if actual != expected_task.upper():
        raise ValueError(
            f"{adapter_dir} has task_type={actual!r}; {expected_task!r} is required. "
            "A sequence-classification adapter cannot perform text extraction."
        )
    return config


def prepare_training_splits(data_dir: str | Path, max_train_samples: int, seed: int):
    train = [dict(row, _source_index=index) for index, row in enumerate(load_split(data_dir, "train"))]
    if max_train_samples:
        random.Random(seed).shuffle(train)
        train = train[:max_train_samples]
    return train, load_split(data_dir, "validation"), load_split(data_dir, "test")


def save_audit_splits(
    output_dir: str | Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    original_data_dir: str | Path | None = None,
) -> tuple[Path, Path | None]:
    output_dir = Path(output_dir)
    audit_dir = output_dir / "audit_data"
    audit_dir.mkdir()
    for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        write_jsonl(audit_dir / f"{split}.jsonl", rows)
    if original_data_dir is None:
        return audit_dir, None

    originals = load_split(original_data_dir, "train")
    largest = max(int(row["_source_index"]) for row in train_rows)
    if largest >= len(originals):
        raise ValueError("Protected and original train splits are not index-aligned")
    aligned = [dict(originals[int(row["_source_index"])], _source_index=row["_source_index"])
               for row in train_rows]
    if any(int(left.get("label", -1)) != int(right.get("label", -2))
           for left, right in zip(train_rows, aligned)):
        raise ValueError("Protected and original train labels are not index-aligned")
    original_dir = output_dir / "audit_original_data"
    original_dir.mkdir()
    for split, rows in (
        ("train", aligned),
        ("validation", load_split(original_data_dir, "validation")),
        ("test", load_split(original_data_dir, "test")),
    ):
        write_jsonl(original_dir / f"{split}.jsonl", rows)
    return audit_dir, original_dir


def normalize_text(text: str) -> str:
    value = html.unescape(unicodedata.normalize("NFKC", str(text))).lower()
    value = re.sub(r"[^\w\s']", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", normalize_text(text), flags=re.UNICODE)


def longest_common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def longest_common_subsequence_length(left: list[Any], right: list[Any]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if left_item == right_item
                           else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def fuzzy_text_metrics(generated: str, reference: str) -> dict[str, Any]:
    generated_normalized = normalize_text(generated)
    reference_normalized = normalize_text(reference)
    generated_tokens = word_tokens(generated)
    reference_tokens = word_tokens(reference)
    overlap = sum((Counter(generated_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lcs = longest_common_subsequence_length(generated_tokens, reference_tokens)
    lcs_precision = lcs / len(generated_tokens) if generated_tokens else 0.0
    lcs_recall = lcs / len(reference_tokens) if reference_tokens else 0.0
    rouge_l = 2 * lcs_precision * lcs_recall / (lcs_precision + lcs_recall) if lcs_precision + lcs_recall else 0.0
    positional = sum(a == b for a, b in zip(generated_tokens, reference_tokens))
    char_similarity = (SequenceMatcher(None, generated_normalized, reference_normalized).ratio()
                       if generated_normalized and reference_normalized else 0.0)
    return {
        "exact": generated == reference,
        "normalized_exact": bool(generated_normalized) and generated_normalized == reference_normalized,
        "generated_tokens": len(generated_tokens),
        "reference_tokens": len(reference_tokens),
        "prefix_match_tokens": longest_common_prefix(generated_tokens, reference_tokens),
        "token_accuracy": positional / len(reference_tokens) if reference_tokens else 0.0,
        "token_f1": token_f1,
        "rouge_l": rouge_l,
        "char_similarity": char_similarity,
        "fuzzy_score": max(token_f1, rouge_l, char_similarity),
    }


def mean(records: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return sum(values) / len(values) if values else 0.0
