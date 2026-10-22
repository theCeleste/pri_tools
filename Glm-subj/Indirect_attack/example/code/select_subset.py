from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "membership",
    "sample_index",
    "source_index",
    "label",
    "text",
    "original_score",
    "neighbor_mean",
    "num_neighbors",
    "score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the fixed SUBJ evaluation subset")
    parser.add_argument("--input", required=True, help="Directory containing full-pool scores")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--member-count", type=int, default=1000)
    parser.add_argument("--nonmember-pool-count", type=int, default=2000)
    parser.add_argument("--nonmember-count", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=60)
    return parser.parse_args()


def read_scores(path: Path, member_count: int, nonmember_pool_count: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    members = frame.loc[frame.membership == "member"].reset_index(drop=True)
    nonmembers = frame.loc[frame.membership == "nonmember"].reset_index(drop=True)
    if (len(members), len(nonmembers)) != (member_count, nonmember_pool_count):
        raise ValueError(
            f"Expected {member_count}+{nonmember_pool_count} rows in {path}, "
            f"got {len(members)}+{len(nonmembers)}"
        )
    return frame


def select_positions(pool_size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0 or count > pool_size:
        raise ValueError(f"Cannot select {count} rows from a pool of {pool_size}")
    if count == pool_size:
        return np.arange(pool_size)
    return np.sort(rng.choice(pool_size, count, replace=False))


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    rng = np.random.default_rng(args.subset_seed)
    member_positions = select_positions(args.member_count, args.member_count, rng)
    nonmember_positions = select_positions(
        args.nonmember_pool_count, args.nonmember_count, rng
    )

    selected_sources: dict[str, list[int]] | None = None
    for variant in args.variants:
        frame = read_scores(
            input_dir / variant / "scores.csv",
            args.member_count,
            args.nonmember_pool_count,
        )
        members = frame.loc[frame.membership == "member"].reset_index(drop=True)
        nonmembers = frame.loc[frame.membership == "nonmember"].reset_index(drop=True)
        selected_member = members.iloc[member_positions].copy()
        selected_nonmember = nonmembers.iloc[nonmember_positions].copy()
        selected_member["sample_index"] = np.arange(len(selected_member))
        selected_nonmember["sample_index"] = np.arange(len(selected_nonmember))
        selected = pd.concat((selected_member, selected_nonmember), ignore_index=True)

        sources = {
            "member_source_indices": selected_member.source_index.astype(int).tolist(),
            "nonmember_source_indices": selected_nonmember.source_index.astype(int).tolist(),
        }
        if selected_sources is None:
            selected_sources = sources
        elif selected_sources != sources:
            raise ValueError("Variants do not contain the same selected samples")
        atomic_csv(selected, output_dir / variant / "scores.csv")

    selection = {
        "subset_seed": args.subset_seed,
        "member_count": args.member_count,
        "nonmember_pool_count": args.nonmember_pool_count,
        "nonmember_count": args.nonmember_count,
        **(selected_sources or {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected {args.member_count} members and {args.nonmember_count} non-members "
        f"with subset seed {args.subset_seed}."
    )


if __name__ == "__main__":
    main()
