from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_evaluator(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "membership": ["member", "member", "nonmember", "nonmember"],
            "score": [0.9, 0.8, 0.2, 0.1],
        }
    ).to_csv(scores, index=False)
    output = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--scores",
            str(scores),
            "--output",
            str(output),
            "--members",
            "2",
            "--nonmembers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result["auc"] == 1.0
    assert result["member_count"] == 2
    assert result["nonmember_count"] == 2


def test_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--mode",
            "full",
            "--method",
            "gradient",
            "--gpu",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare_neighbors.py" in result.stdout
    assert "weights/mezo" in result.stdout
    assert "weights/zo_dpga" in result.stdout
    assert "weights/dpzero" not in result.stdout


def test_example_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--mode",
            "full",
            "--method",
            "example",
            "--gpu",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--member-count 1000" in result.stdout
    assert "--nonmember-count 2000" in result.stdout
    assert "--seed 43" in result.stdout
    assert "select_subset.py" in result.stdout
    assert "--subset-seed 60" in result.stdout


def test_example_subset_selection(tmp_path: Path) -> None:
    full_pool = tmp_path / "full_pool"
    output = tmp_path / "selected"
    rows = []
    for membership, count, offset in (("member", 2, 0), ("nonmember", 4, 100)):
        for index in range(count):
            rows.append(
                {
                    "membership": membership,
                    "sample_index": index,
                    "source_index": offset + index,
                    "label": index % 2,
                    "text": f"{membership}-{index}",
                    "original_score": float(index),
                    "neighbor_mean": 0.0,
                    "num_neighbors": 10,
                    "score": float(index),
                }
            )
    for variant in ("mean", "private"):
        destination = full_pool / variant
        destination.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(destination / "scores.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "example/code/select_subset.py"),
            "--input",
            str(full_pool),
            "--output",
            str(output),
            "--variants",
            "mean",
            "private",
            "--member-count",
            "2",
            "--nonmember-pool-count",
            "4",
            "--nonmember-count",
            "2",
            "--subset-seed",
            "60",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mean = pd.read_csv(output / "mean/scores.csv")
    private = pd.read_csv(output / "private/scores.csv")
    assert (mean.membership == "member").sum() == 2
    assert (mean.membership == "nonmember").sum() == 2
    assert mean.source_index.tolist() == private.source_index.tolist()
