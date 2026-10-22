from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET_FPRS = (1.0, 0.5, 0.1, 0.01, 0.001, 0.0001)


def atomic_csv(frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def rate_at_threshold(member: np.ndarray, nonmember: np.ndarray, threshold: float):
    member_prediction = member > threshold
    nonmember_prediction = nonmember > threshold
    tp = int(member_prediction.sum())
    fp = int(nonmember_prediction.sum())
    tn = len(nonmember) - fp
    fn = len(member) - tp
    return tp / len(member), fp / len(nonmember), tp, fp, tn, fn


def best_at_fpr(member: np.ndarray, nonmember: np.ndarray, target_fpr: float):
    all_scores = np.concatenate([member, nonmember])
    epsilon = max(1e-12, float(np.max(np.abs(all_scores))) * 1e-12)
    candidates = np.unique(
        np.concatenate([all_scores, [all_scores.min() - epsilon, all_scores.max() + epsilon]])
    )
    best = None
    for threshold in candidates:
        tpr, fpr, tp, fp, tn, fn = rate_at_threshold(member, nonmember, float(threshold))
        candidate = (float(threshold), tpr, fpr, tp, fp, tn, fn)
        if fpr <= target_fpr and (
            best is None or tpr > best[1] or (tpr == best[1] and threshold < best[0])
        ):
            best = candidate
    if best is None:
        raise RuntimeError("No threshold satisfies the requested FPR")
    return best


def evaluate_scores(score_csv: str | Path, output_dir: str | Path, members: int, nonmembers: int) -> dict:
    frame = pd.read_csv(score_csv)
    if frame["score"].isna().any() or not np.isfinite(frame["score"].to_numpy()).all():
        raise ValueError("Score file contains NaN or Inf")
    member = frame.loc[frame.membership == "member", "score"].to_numpy(float)
    nonmember = frame.loc[frame.membership == "nonmember", "score"].to_numpy(float)
    if (len(member), len(nonmember)) != (members, nonmembers):
        raise ValueError(
            f"Refusing partial evaluation: expected {members}+{nonmembers}, "
            f"got {len(member)}+{len(nonmember)}"
        )
    auc = float(roc_auc_score(np.r_[np.ones(len(member)), np.zeros(len(nonmember))], np.r_[member, nonmember]))
    rows = []
    for target in TARGET_FPRS:
        threshold, tpr, actual_fpr, tp, fp, tn, fn = best_at_fpr(member, nonmember, target)
        rows.append(
            {
                "target_fpr": target,
                "target_fpr_percent": f"{target * 100:.4f}%",
                "actual_fpr": actual_fpr,
                "actual_fpr_percent": f"{actual_fpr * 100:.4f}%",
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(rows), output_dir / "metrics.csv")
    summary = {
        "auc": auc,
        "member_count": len(member),
        "nonmember_count": len(nonmember),
        "member_score_mean": float(member.mean()),
        "nonmember_score_mean": float(nonmember.mean()),
        "metrics": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
