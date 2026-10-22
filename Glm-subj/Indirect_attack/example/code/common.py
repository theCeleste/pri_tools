from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET_FPRS = (1.0, 0.5, 0.1, 0.01, 0.001, 0.0001)


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def validate_member_nonmember(
    members: Sequence[dict],
    nonmembers: Sequence[dict],
    member_count: int,
    nonmember_count: int,
) -> None:
    if len(members) != member_count or len(nonmembers) != nonmember_count:
        raise ValueError(
            f"Expected {member_count}+{nonmember_count}, got "
            f"{len(members)}+{len(nonmembers)}"
        )
    member_keys = [normalize_text(row["text"]) for row in members]
    nonmember_keys = [normalize_text(row["text"]) for row in nonmembers]
    if len(member_keys) != len(set(member_keys)):
        raise ValueError("Member attack samples contain duplicates")
    if len(nonmember_keys) != len(set(nonmember_keys)):
        raise ValueError("Non-member attack samples contain duplicates")
    overlap = set(member_keys) & set(nonmember_keys)
    if overlap:
        raise ValueError(f"Member/non-member overlap: {len(overlap)} texts")


def atomic_csv(frame: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def load_checkpoint(path: str | Path) -> dict[tuple[str, int], dict]:
    path = Path(path)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    return {
        (str(row["membership"]), int(row["sample_index"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def save_checkpoint(records: Iterable[dict], destination: str | Path) -> None:
    frame = pd.DataFrame(list(records))
    if not frame.empty:
        frame = frame.sort_values(["membership", "sample_index"]).reset_index(drop=True)
    atomic_csv(frame, destination)


def _rate_at_threshold(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, int, int, int, int]:
    member_prediction = member_scores > threshold
    nonmember_prediction = nonmember_scores > threshold
    tp = int(member_prediction.sum())
    fn = int((~member_prediction).sum())
    fp = int(nonmember_prediction.sum())
    tn = int((~nonmember_prediction).sum())
    return tp / len(member_scores), fp / len(nonmember_scores), tp, fp, tn, fn


def _best_at_fpr(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    target_fpr: float,
) -> tuple[float, float, float, int, int, int, int]:
    all_scores = np.concatenate([member_scores, nonmember_scores])
    epsilon = max(1e-12, float(np.max(np.abs(all_scores))) * 1e-12)
    candidates = np.unique(
        np.concatenate(
            [all_scores, [np.min(all_scores) - epsilon, np.max(all_scores) + epsilon]]
        )
    )
    best = None
    for threshold in candidates:
        tpr, fpr, tp, fp, tn, fn = _rate_at_threshold(
            member_scores, nonmember_scores, float(threshold)
        )
        candidate = (float(threshold), tpr, fpr, tp, fp, tn, fn)
        if fpr <= target_fpr and (
            best is None or tpr > best[1] or (tpr == best[1] and threshold < best[0])
        ):
            best = candidate
    if best is None:
        raise RuntimeError("No threshold satisfies the requested FPR")
    return best


def evaluate_scores(
    score_csv: str | Path,
    output_dir: str | Path,
    expected_members: int,
    expected_nonmembers: int,
) -> dict:
    frame = pd.read_csv(score_csv)
    required = {"membership", "score"}
    if not required <= set(frame.columns):
        raise ValueError(f"Score CSV is missing columns: {required - set(frame.columns)}")
    if frame["score"].isna().any() or not np.isfinite(frame["score"].to_numpy()).all():
        raise ValueError("Score CSV contains NaN or Inf")
    member = frame.loc[frame.membership == "member", "score"].to_numpy(dtype=float)
    nonmember = frame.loc[frame.membership == "nonmember", "score"].to_numpy(dtype=float)
    if len(member) != expected_members or len(nonmember) != expected_nonmembers:
        raise ValueError(
            f"Refusing partial evaluation: expected {expected_members}+{expected_nonmembers}, "
            f"got {len(member)}+{len(nonmember)}"
        )
    labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    scores = np.concatenate([member, nonmember])
    auc = float(roc_auc_score(labels, scores))
    rows = []
    for target in TARGET_FPRS:
        threshold, tpr, fpr, tp, fp, tn, fn = _best_at_fpr(member, nonmember, target)
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
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def ensure_finite(values: Sequence[float], context: str) -> None:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"Non-finite or empty scores for {context}")
