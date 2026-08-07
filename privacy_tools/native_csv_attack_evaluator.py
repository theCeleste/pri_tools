from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_scores(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "diff_score" not in reader.fieldnames:
            raise ValueError(f"{path} 缺少 diff_score 列")
        scores: list[float] = []
        for row in reader:
            raw = row.get("diff_score", "")
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scores.append(score)
    if not scores:
        raise ValueError(f"{path} 没有可用的 diff_score")
    return scores


def pairwise_auc(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def threshold_at_fpr(negative: list[float], target: float) -> float:
    # 与原脚本的 ROC 逻辑一致：分数越大越可能是成员，并选择 FPR 不超过目标值的阈值。
    threshold = math.inf
    for candidate in sorted(set(negative), reverse=True):
        observed = sum(score >= candidate for score in negative) / len(negative)
        if observed <= target:
            threshold = candidate
        else:
            break
    return threshold


def main() -> int:
    parser = argparse.ArgumentParser(description="评估原生邻居成员推断脚本新生成的成员/非成员得分")
    parser.add_argument("--member-csv", type=Path, required=True)
    parser.add_argument("--nonmember-csv", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.01, 0.001])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    member = read_scores(args.member_csv)
    nonmember = read_scores(args.nonmember_csv)
    rows: list[dict[str, object]] = []
    for target in args.target_fprs:
        if not 0 < target < 1:
            raise ValueError("target FPR 必须在 0 和 1 之间")
        threshold = threshold_at_fpr(nonmember, target)
        tpr = sum(score >= threshold for score in member) / len(member)
        fpr = sum(score >= threshold for score in nonmember) / len(nonmember)
        rows.append({
            "target_fpr": target,
            "threshold": threshold if math.isfinite(threshold) else None,
            "tpr": tpr,
            "observed_fpr": fpr,
            "fpr_resolution": 1.0 / len(nonmember),
            "target_statistically_resolved": target >= 1.0 / len(nonmember),
        })

    summary = {
        "schema_version": 1,
        "evaluation": "native_neighbour_membership_inference",
        "method": args.method,
        "fresh_scores_generated": True,
        "native_attack_generator": True,
        "member_samples": len(member),
        "nonmember_samples": len(nonmember),
        "auc": pairwise_auc(member, nonmember),
        "rows": rows,
        "member_csv": str(args.member_csv),
        "nonmember_csv": str(args.nonmember_csv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
