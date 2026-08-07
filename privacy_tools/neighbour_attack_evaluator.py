from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def score_gaps(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    gaps: list[float] = []
    for record in records:
        original_key = next((key for key in record if isinstance(key, str) and key.startswith("<original_text>:")), None)
        if original_key is None:
            continue
        neighbour_scores: list[float] = []
        for key, score in record.items():
            if key == original_key or not (isinstance(key, tuple) and len(key) == 2):
                continue
            value = score[0] if isinstance(score, list) and score else score
            try:
                neighbour_scores.append(float(value))
            except (TypeError, ValueError):
                continue
        if neighbour_scores:
            gaps.append(float(record[original_key]) - float(np.mean(neighbour_scores)))
    return np.asarray(gaps, dtype=float)


def subsample(positive: np.ndarray, negative: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if ratio <= 0:
        raise ValueError("--pos-neg-ratio 必须大于 0")
    negative_count = min(len(negative), int(len(positive) / ratio))
    positive_count = min(len(positive), int(round(negative_count * ratio)))
    if positive_count < 1 or negative_count < 1:
        raise ValueError("当前正负样本数量不足以按指定比例评估")
    rng = np.random.RandomState(seed)
    return (
        positive[rng.choice(len(positive), positive_count, replace=False)],
        negative[rng.choice(len(negative), negative_count, replace=False)],
    )


def rates(positive: np.ndarray, negative: np.ndarray, threshold: float) -> tuple[float, float]:
    return float(np.mean(positive > threshold)), float(np.mean(negative > threshold))


def main() -> int:
    parser = argparse.ArgumentParser(description="复核共享遗留邻居成员推断样例")
    parser.add_argument("--base-train-pkl", type=Path, required=True)
    parser.add_argument("--base-test-pkl", type=Path, required=True)
    parser.add_argument("--priv-train-pkl", type=Path, required=True)
    parser.add_argument("--priv-test-pkl", type=Path, required=True)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.01, 0.001, 0.0001])
    parser.add_argument("--pos-neg-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", default="shared_legacy_reference")
    parser.add_argument("--method-specific", action="store_true")
    args = parser.parse_args()

    base_train, base_test = subsample(
        score_gaps(args.base_train_pkl), score_gaps(args.base_test_pkl), args.pos_neg_ratio, args.seed
    )
    protected_train, protected_test = subsample(
        score_gaps(args.priv_train_pkl), score_gaps(args.priv_test_pkl), args.pos_neg_ratio, args.seed + 1
    )
    rows: list[dict[str, object]] = []
    for target in args.target_fprs:
        # 阈值只由基线非成员分数确定；随后固定阈值比较基线和受保护结果。
        threshold = float(np.quantile(base_test, 1.0 - target, method="higher"))
        base_tpr, base_fpr = rates(base_train, base_test, threshold)
        protected_tpr, protected_fpr = rates(protected_train, protected_test, threshold)
        rows.append({
            "target_fpr": target,
            "threshold_from_baseline": threshold,
            "baseline_tpr": base_tpr,
            "baseline_fpr": base_fpr,
            "protected_tpr": protected_tpr,
            "protected_fpr": protected_fpr,
            "positive_samples": len(base_train),
            "negative_samples": len(base_test),
            "scope": args.scope,
            "method_independent_evidence": args.method_specific,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "evaluation": "neighbour_membership_inference",
        "scope": args.scope,
        "method_independent_evidence": args.method_specific,
        "rows": rows,
        "csv": str(args.output),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.method_specific:
        print("注意：这是共享遗留样例，不构成任何单一方法的独立攻击结论。")
    return 0
    print("注意：这是共享遗留样例，不构成任何单一方法的独立攻击结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
