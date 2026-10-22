from __future__ import annotations

import argparse
import json
import os
import random
from heapq import nlargest
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_utils import load_splits, write_jsonl


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare fixed BERT neighbors for SUBJ")
    parser.add_argument("--splits", default=str(local / "data/splits"))
    parser.add_argument(
        "--proposal-model",
        default=str(root / "resources/bert-base-uncased"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proposal-max-length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--neighbor-top-k", type=int, default=6)
    parser.add_argument("--candidate-neighbors", type=int, default=50)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--member-limit", type=int)
    parser.add_argument("--nonmember-limit", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate_neighbors(text, tokenizer, model, device, args) -> list[str]:
    ids = tokenizer(
        text,
        truncation=True,
        max_length=args.proposal_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    replacements = {}
    dropout = torch.nn.Dropout(p=args.dropout)
    backbone = getattr(model, model.base_model_prefix)
    for position in range(1, ids.shape[1]):
        original = ids[0, position]
        embeddings = backbone.embeddings(ids)
        embeddings = torch.cat(
            (
                embeddings[:, :position],
                dropout(embeddings[:, position]).unsqueeze(1),
                embeddings[:, position + 1 :],
            ),
            dim=1,
        )
        probabilities = torch.softmax(model(inputs_embeds=embeddings).logits, dim=2)
        original_probability = probabilities[0, position, original].item()
        values, candidates = torch.topk(probabilities[:, position, :], args.neighbor_top_k, dim=1)
        denominator = 0.1 if original_probability == 1 else 1 - original_probability
        for candidate, value in zip(candidates[0], values[0]):
            if candidate != original:
                replacements[(position, int(candidate))] = value.item() / denominator
    result = []
    for position, candidate in nlargest(args.candidate_neighbors, replacements, key=replacements.get):
        alternate = ids.clone()
        alternate[0, position] = candidate
        decoded = tokenizer.batch_decode(alternate)[0]
        decoded = decoded.replace(" [SEP]", " ").replace("[CLS] ", " ").strip()
        if decoded and decoded not in result:
            result.append(decoded)
    return result[: args.num_neighbors]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    members, _, nonmembers = load_splits(args.splits)
    if args.member_limit is not None:
        members = members[: args.member_limit]
    if args.nonmember_limit is not None:
        nonmembers = nonmembers[: args.nonmember_limit]
    samples = [("member", index, row) for index, row in enumerate(members)]
    samples.extend(("nonmember", index, row) for index, row in enumerate(nonmembers))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.proposal_model)
    device = torch.device(args.device)
    model = AutoModelForMaskedLM.from_pretrained(args.proposal_model).to(device).eval()
    completed = []
    failures = []
    for membership, sample_index, row in tqdm(samples, desc="Preparing fixed neighbors"):
        try:
            neighbors = generate_neighbors(row["text"], tokenizer, model, device, args)
            if len(neighbors) < args.num_neighbors:
                raise ValueError(f"Only generated {len(neighbors)} neighbors")
            completed.append(
                {
                    "membership": membership,
                    "sample_index": sample_index,
                    "source_index": row["source_index"],
                    "text": row["text"],
                    "neighbors": neighbors,
                }
            )
        except Exception as exc:
            failures.append(
                {"membership": membership, "sample_index": sample_index, "error": str(exc)}
            )
        if len(completed) % args.checkpoint_every == 0:
            write_jsonl(completed, output)
    write_jsonl(completed, output)
    config = vars(args).copy()
    config.update(
        {
            "splits": args.splits,
            "proposal_model": args.proposal_model,
            "output": args.output,
            "member_count": len(members),
            "nonmember_count": len(nonmembers),
            "row_count": len(completed),
            "failures": failures,
        }
    )
    output.with_suffix(output.suffix + ".config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if failures or len(completed) != len(samples):
        raise RuntimeError(f"Neighbor generation incomplete: {len(completed)}/{len(samples)}")
    print(json.dumps({"output": str(output), "rows": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
