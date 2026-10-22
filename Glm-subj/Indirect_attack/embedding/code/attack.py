from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from data_utils import load_splits, read_jsonl
from metrics import atomic_csv, evaluate_scores
from modeling import load_adapter_model, load_tokenizer, sequence_log_probabilities


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Indirect MIA for a MeZO/DPZero ChatGLM adapter")
    parser.add_argument("--model", default=str(root.parent / "models/chatglm3-6b"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--splits", default=str(local / "data/splits"))
    parser.add_argument("--neighbors", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--loss-prefix-tokens", type=int, default=0)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--member-limit", type=int)
    parser.add_argument("--nonmember-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members, _, nonmembers = load_splits(args.splits)
    if args.member_limit is not None:
        members = members[: args.member_limit]
    if args.nonmember_limit is not None:
        nonmembers = nonmembers[: args.nonmember_limit]
    samples = [("member", index, row) for index, row in enumerate(members)]
    samples.extend(("nonmember", index, row) for index, row in enumerate(nonmembers))
    neighbor_rows = read_jsonl(args.neighbors)
    neighbors = {
        (row["membership"], int(row["sample_index"])): row["neighbors"]
        for row in neighbor_rows
    }
    expected_keys = {(membership, index) for membership, index, _ in samples}
    if set(neighbors) != expected_keys:
        raise ValueError("Neighbor cache does not exactly match the attack split")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "scores.csv"
    completed = {}
    if args.resume and score_path.is_file():
        prior = pd.read_csv(score_path)
        completed = {
            (str(row.membership), int(row.sample_index)): row._asdict()
            for row in prior.itertuples(index=False)
        }
    tokenizer = load_tokenizer(args.model)
    model = load_adapter_model(args.model, args.adapter, args.device, args.dtype)
    failures = []
    for membership, sample_index, row in tqdm(samples, desc="Indirect MIA"):
        key = (membership, sample_index)
        if key in completed:
            continue
        error = None
        for _ in range(2):
            try:
                texts = [row["text"], *neighbors[key]]
                values = sequence_log_probabilities(
                    model,
                    tokenizer,
                    texts,
                    max_length=args.max_length,
                    batch_size=args.score_batch_size,
                    loss_prefix_tokens=args.loss_prefix_tokens,
                )
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError("Non-finite target-model score")
                neighbor_mean = float(np.mean(values[1:]))
                completed[key] = {
                    "membership": membership,
                    "sample_index": sample_index,
                    "source_index": row["source_index"],
                    "label": row["label"],
                    "text": row["text"],
                    "original_score": values[0],
                    "neighbor_mean": neighbor_mean,
                    "num_neighbors": len(values) - 1,
                    "score": values[0] - neighbor_mean,
                }
                error = None
                break
            except Exception as exc:
                error = str(exc)
                torch.cuda.empty_cache()
        if error is not None:
            failures.append(
                {"membership": membership, "sample_index": sample_index, "error": error}
            )
        if len(completed) % args.checkpoint_every == 0 or error is not None:
            frame = pd.DataFrame(completed.values()).sort_values(["membership", "sample_index"])
            atomic_csv(frame, score_path)

    frame = pd.DataFrame(completed.values()).sort_values(["membership", "sample_index"])
    atomic_csv(frame, score_path)
    (output / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config = vars(args).copy()
    config.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "splits": args.splits,
            "neighbors": args.neighbors,
            "member_count": len(members),
            "nonmember_count": len(nonmembers),
        }
    )
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failures or len(completed) != len(samples):
        raise RuntimeError(
            f"Attack incomplete: {len(completed)}/{len(samples)}, failures={len(failures)}"
        )
    summary = evaluate_scores(score_path, output / "evaluation", len(members), len(nonmembers))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
