from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def load_samples(path: Path, legacy: bool) -> list[str]:
    if not legacy:
        samples: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{path}:{line_no} 缺少字符串字段 text")
                samples.append(text)
        return samples

    # 遗留文件按“排名行 + 空行 + 样本文本”分块；同一样本会在多个指标下重复出现。
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?m)^\d+:\s+[^\n]+\n\s*\n", content)[1:]
    return [block.split("\n\n", 1)[0].strip() for block in blocks if block.strip()]


def load_training_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"训练 JSON 必须是列表：{path}")
    texts: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        value = next(
            (item.get(key) for key in ("original_text", "text", "sentence") if isinstance(item.get(key), str)),
            None,
        )
        if value is None and isinstance(item.get("question"), str):
            match = re.search(r'Sentence:\s*["“](.*?)["”]', item["question"], flags=re.DOTALL)
            value = match.group(1) if match else item["question"]
        if value:
            texts.append(value)
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description="严格评估抽取样本与训练语句的重合")
    parser.add_argument("--samples", type=Path, required=True)
    training_group = parser.add_mutually_exclusive_group(required=True)
    training_group.add_argument("--train-tsv", type=Path)
    training_group.add_argument("--train-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-format", action="store_true")
    parser.add_argument(
        "--scope",
        choices=(
            "method_run",
            "shared_legacy_reference",
            "ptcds_method_specific_reference",
            "ptcds_method_specific_attack",
            "protected",
            "baseline",
        ),
        default="method_run",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples, args.legacy_format)
    if args.train_tsv is not None:
        with args.train_tsv.open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            training_texts = [row["sentence"] for row in rows if row.get("sentence")]
    else:
        training_texts = load_training_json(args.train_json)
    training = {normalize(text): text for text in training_texts if normalize(text)}

    unique_samples = {normalize(sample): sample for sample in samples if normalize(sample)}
    full_matches = sorted(set(unique_samples) & set(training))
    # 行匹配只接受完整非空行；不再把标点拆出的单字符或指标数字计入分母。
    unique_lines: dict[str, str] = {}
    for sample in unique_samples.values():
        for line in sample.replace("<0x0A>", "\n").splitlines():
            key = normalize(line)
            if len(key) >= 3:
                unique_lines[key] = line.strip()
    line_matches = sorted(set(unique_lines) & set(training))

    result = {
        "schema_version": 1,
        "evaluation": "strict_training_overlap",
        "scope": args.scope,
        "method_independent_evidence": args.scope != "shared_legacy_reference",
        "source_file": str(args.samples.resolve()),
        "parsed_records": len(samples),
        "unique_samples": len(unique_samples),
        "full_sample_matches": len(full_matches),
        "full_sample_match_rate": len(full_matches) / len(unique_samples) if unique_samples else 0.0,
        "unique_complete_lines": len(unique_lines),
        "complete_line_matches": len(line_matches),
        "complete_line_match_rate": len(line_matches) / len(unique_lines) if unique_lines else 0.0,
        "matched_training_sentences": [training[key] for key in line_matches[:100]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.scope == "shared_legacy_reference":
        print("注意：这是多个方法目录曾重复携带的共享遗留样例，不构成任何单一方法的独立攻击结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
