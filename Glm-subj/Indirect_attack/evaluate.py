from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET_FPRS = (1.0, 0.5, 0.1, 0.01, 0.001, 0.0001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate indirect MIA scores")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--members", type=int, required=True)
    parser.add_argument("--nonmembers", type=int, required=True)
    return parser.parse_args()


def rates(member: np.ndarray, nonmember: np.ndarray, threshold: float):
    member_positive = member > threshold
    nonmember_positive = nonmember > threshold
    tp = int(member_positive.sum())
    fp = int(nonmember_positive.sum())
    return tp / len(member), fp / len(nonmember), tp, fp, len(nonmember) - fp, len(member) - tp


def best_at_fpr(member: np.ndarray, nonmember: np.ndarray, target: float):
    scores = np.concatenate((member, nonmember))
    epsilon = max(1e-12, float(np.max(np.abs(scores))) * 1e-12)
    candidates = np.unique(np.concatenate((scores, [scores.min() - epsilon, scores.max() + epsilon])))
    best = None
    for threshold in candidates:
        tpr, fpr, tp, fp, tn, fn = rates(member, nonmember, float(threshold))
        candidate = (float(threshold), tpr, fpr, tp, fp, tn, fn)
        if fpr <= target and (best is None or tpr > best[1] or (tpr == best[1] and threshold < best[0])):
            best = candidate
    if best is None:
        raise RuntimeError("No threshold satisfies target FPR")
    return best


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scores)
    if not {"membership", "score"} <= set(frame.columns):
        raise ValueError("scores.csv must contain membership and score columns")
    if frame.score.isna().any() or not np.isfinite(frame.score.to_numpy()).all():
        raise ValueError("scores.csv contains NaN or Inf")
    member = frame.loc[frame.membership == "member", "score"].to_numpy(float)
    nonmember = frame.loc[frame.membership == "nonmember", "score"].to_numpy(float)
    if (len(member), len(nonmember)) != (args.members, args.nonmembers):
        raise ValueError(
            f"Expected {args.members}+{args.nonmembers}, got {len(member)}+{len(nonmember)}"
        )
    auc = float(
        roc_auc_score(
            np.r_[np.ones(len(member)), np.zeros(len(nonmember))],
            np.r_[member, nonmember],
        )
    )
    rows = []
    for target in TARGET_FPRS:
        threshold, tpr, fpr, tp, fp, tn, fn = best_at_fpr(member, nonmember, target)
        rows.append(
            {
                "target_fpr": target,
                "target_fpr_percent": f"{target * 100:.4f}%",
                "actual_fpr": fpr,
                "actual_fpr_percent": f"{fpr * 100:.4f}%",
                "tpr": tpr,
                "tpr_percent": f"{tpr * 100:.4f}%",
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "auc": auc,
                "member_count": len(member),
                "nonmember_count": len(nonmember),
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    summary = {
        "auc": auc,
        "member_count": len(member),
        "nonmember_count": len(nonmember),
        "member_score_mean": float(member.mean()),
        "nonmember_score_mean": float(nonmember.mean()),
        "metrics": rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
