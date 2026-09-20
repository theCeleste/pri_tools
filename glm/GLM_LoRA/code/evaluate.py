import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def load_scores(path: str) -> List[Dict[Any, Any]]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def first_scalar(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_single_text(scores_dict: Dict[Any, Any]) -> Optional[Dict[str, Any]]:
    original_key = None
    original_score = None
    for key, value in scores_dict.items():
        if isinstance(key, str) and key.startswith("<original_text>:"):
            original_key = key
            original_score = first_scalar(value)
            break

    if original_key is None or original_score is None:
        return None

    neighbour_scores: List[float] = []
    for key, value in scores_dict.items():
        if key == original_key:
            continue
        score = first_scalar(value)
        if score is not None:
            neighbour_scores.append(score)

    if not neighbour_scores:
        return None

    neighbour_array = np.asarray(neighbour_scores, dtype=np.float64)
    neighbour_mean = float(np.mean(neighbour_array))
    return {
        "original_text": original_key.replace("<original_text>: ", "", 1),
        "original_score": float(original_score),
        "num_neighbours": int(len(neighbour_scores)),
        "neighbour_scores_mean": neighbour_mean,
        "neighbour_scores_std": float(np.std(neighbour_array)),
        "neighbour_scores_max": float(np.max(neighbour_array)),
        "neighbour_scores_min": float(np.min(neighbour_array)),
        "score_gap": float(original_score - neighbour_mean),
        "score_gap_max": float(original_score - np.max(neighbour_array)),
    }


def extract_score_gaps(all_scores: Iterable[Dict[Any, Any]]) -> np.ndarray:
    gaps = []
    skipped = 0
    for scores_dict in all_scores:
        analysis = analyze_single_text(scores_dict)
        if analysis is None:
            skipped += 1
            continue
        gaps.append(analysis["score_gap"])

    if skipped:
        print(f"Skipped samples without valid neighbours: {skipped}")
    return np.asarray(gaps, dtype=np.float64)


def rate_at_threshold(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float, int, int, int, int]:
    member_pred = member_scores > threshold
    nonmember_pred = nonmember_scores > threshold

    tp = int(member_pred.sum())
    fn = int((~member_pred).sum())
    fp = int(nonmember_pred.sum())
    tn = int((~nonmember_pred).sum())

    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return tpr, fpr, tp, fp, tn, fn


def best_threshold_for_fpr(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    target_fpr: float,
) -> Tuple[float, float, float, int, int, int, int]:
    all_scores = np.concatenate([member_scores, nonmember_scores])
    if all_scores.size == 0:
        raise ValueError("No score gaps available.")

    epsilon = max(1e-12, abs(float(np.max(all_scores))) * 1e-12)
    candidates = np.unique(np.concatenate([all_scores, [np.max(all_scores) + epsilon]]))

    best = None
    for threshold in candidates:
        tpr, fpr, tp, fp, tn, fn = rate_at_threshold(member_scores, nonmember_scores, threshold)
        if fpr <= target_fpr:
            if best is None or tpr > best[1] or (tpr == best[1] and threshold < best[0]):
                best = (float(threshold), tpr, fpr, tp, fp, tn, fn)

    if best is None:
        threshold = float(np.max(all_scores) + epsilon)
        tpr, fpr, tp, fp, tn, fn = rate_at_threshold(member_scores, nonmember_scores, threshold)
        best = (threshold, tpr, fpr, tp, fp, tn, fn)

    return best


def describe(name: str, scores: np.ndarray) -> None:
    print(f"{name}:")
    print(f"  count: {len(scores)}")
    print(f"  mean : {np.mean(scores):.6f}")
    print(f"  std  : {np.std(scores):.6f}")
    print(f"  min  : {np.min(scores):.6f}")
    print(f"  max  : {np.max(scores):.6f}")


def evaluate(
    member_pkl: str,
    nonmember_pkl: str,
    target_fprs: Sequence[float],
    output_csv: str,
) -> pd.DataFrame:
    print("=" * 80)
    print("Neighbour MIA evaluation")
    print("=" * 80)
    print(f"Member pkl   : {member_pkl}")
    print(f"Nonmember pkl: {nonmember_pkl}")

    member_scores = extract_score_gaps(load_scores(member_pkl))
    nonmember_scores = extract_score_gaps(load_scores(nonmember_pkl))

    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        raise RuntimeError("Both member and nonmember pkls need at least one valid scored sample.")

    print()
    describe("Member score_gap", member_scores)
    describe("Nonmember score_gap", nonmember_scores)

    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    y_score = np.concatenate([member_scores, nonmember_scores])
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)
    auc_score = float(auc(fpr_curve, tpr_curve))

    print()
    print(f"AUC-ROC: {auc_score:.6f}")
    print()
    print("TPR at target FPR")
    print("-" * 80)

    rows = []
    for target_fpr in target_fprs:
        threshold, tpr, fpr, tp, fp, tn, fn = best_threshold_for_fpr(
            member_scores,
            nonmember_scores,
            target_fpr,
        )
        row = {
            "target_fpr": float(target_fpr),
            "target_fpr_percent": f"{target_fpr * 100:.4f}%",
            "threshold": threshold,
            "actual_fpr": fpr,
            "actual_fpr_percent": f"{fpr * 100:.4f}%",
            "tpr": tpr,
            "tpr_percent": f"{tpr * 100:.4f}%",
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "auc": auc_score,
            "member_count": int(len(member_scores)),
            "nonmember_count": int(len(nonmember_scores)),
        }
        rows.append(row)
        print(
            f"target_fpr={target_fpr:.6g} "
            f"actual_fpr={fpr:.6g} "
            f"tpr={tpr:.6g} "
            f"threshold={threshold:.6f} "
            f"tp/fp/tn/fn={tp}/{fp}/{tn}/{fn}"
        )

    frame = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print()
    print(f"Saved evaluation CSV: {output}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neighbour-based MIA pkls.")
    parser.add_argument("--member-pkl", "--train_pkl", dest="member_pkl", required=True)
    parser.add_argument("--nonmember-pkl", "--test_pkl", dest="nonmember_pkl", required=True)
    parser.add_argument(
        "--target-fprs",
        "--target_fprs",
        dest="target_fprs",
        type=float,
        nargs="+",
        default=[0.01, 0.001, 0.0001],
    )
    parser.add_argument("--output-csv", "--output_csv", dest="output_csv", default="mia_evaluation_results.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(args.member_pkl, args.nonmember_pkl, args.target_fprs, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
