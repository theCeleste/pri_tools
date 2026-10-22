from __future__ import annotations

import argparse
from heapq import nlargest
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common import (
    ensure_finite,
    evaluate_scores,
    load_checkpoint,
    read_json,
    read_jsonl,
    save_checkpoint,
    set_seed,
    validate_member_nonmember,
)


QUESTION_PATTERN = re.compile(r'Sentence: "(.*)"\n\nOptions:', re.DOTALL)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="SUBJ Mean/Private Steering indirect MIA")
    parser.add_argument("--model", default=str(root.parent / "models/chatglm3-6b"))
    parser.add_argument(
        "--interface-dir", default=str(root / "example/interface")
    )
    parser.add_argument(
        "--proposal-model", default=str(root / "resources/bert-base-uncased")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--member-count", type=int, default=10)
    parser.add_argument("--nonmember-count", type=int, default=500)
    parser.add_argument("--member-start", type=int, default=0)
    parser.add_argument("--nonmember-start", type=int, default=0)
    parser.add_argument("--construction-samples", type=int, default=1000)
    parser.add_argument("--construction-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--proposal-max-length", type=int, default=512)
    parser.add_argument("--neighbor-top-k", type=int, default=6)
    parser.add_argument("--candidate-neighbors", type=int, default=50)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument("--read-token-index", type=int, default=-2)
    parser.add_argument("--clip", type=float, default=18.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.005)
    parser.add_argument("--steering-multiplier", type=float, default=256.0)
    parser.add_argument("--private-seed", type=int, default=42)
    parser.add_argument("--variants", choices=("both", "mean", "private"), default="both")
    parser.add_argument("--neighbor-cache")
    parser.add_argument("--utility-start", type=int, default=0)
    parser.add_argument("--utility-count", type=int, default=0)
    parser.add_argument("--utility-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--construction-limit", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_record(record: dict, index: int) -> dict:
    match = QUESTION_PATTERN.search(record["question"])
    if not match:
        raise ValueError(f"Cannot extract SUBJ sentence from row {index}")
    answer = record["answer_matching_behavior"]
    label = 1 if "subjective" in answer.lower() else 0
    return {"source_index": index, "text": match.group(1), "label": label}


def load_samples(
    interface_dir: Path,
    member_count: int,
    nonmember_count: int,
    *,
    member_start: int = 0,
    nonmember_start: int = 0,
    construction_samples: int = 1000,
    construction_seed: int = 42,
):
    if min(member_count, nonmember_count, member_start, nonmember_start) < 0:
        raise ValueError("Sample counts and starts must be non-negative")
    if construction_samples <= 0:
        raise ValueError("construction-samples must be greater than zero")
    construction = read_json(interface_dir / "data/generate_dataset.json")
    indexed = list(enumerate(construction))
    random.Random(construction_seed).shuffle(indexed)
    selected = indexed[:construction_samples]
    member_stop = member_start + member_count
    if member_stop > len(selected):
        raise ValueError(
            f"Member slice [{member_start}:{member_stop}] exceeds "
            f"{len(selected)} construction samples"
        )
    members = [
        parse_record(record, index) for index, record in selected[member_start:member_stop]
    ]
    test = read_json(interface_dir / "data/test_dataset_ab.json")
    nonmember_stop = nonmember_start + nonmember_count
    if nonmember_stop > len(test):
        raise ValueError(
            f"Non-member slice [{nonmember_start}:{nonmember_stop}] exceeds "
            f"the {len(test)}-row test pool"
        )
    nonmembers = [
        parse_record(test[index], index) for index in range(nonmember_start, nonmember_stop)
    ]
    for row in members:
        row["membership"] = "member"
    for row in nonmembers:
        row["membership"] = "nonmember"
    validate_member_nonmember(members, nonmembers, member_count, nonmember_count)
    return members, nonmembers, construction


def load_cached_neighbors(path: str | Path, samples: list[tuple[str, int, dict]], args):
    path = Path(path).resolve()
    config_path = path.with_suffix(path.suffix + ".config.json")
    if not config_path.is_file():
        raise ValueError(f"Neighbor cache metadata is missing: {config_path}")
    config = read_json(config_path)
    expected = {
        "seed": args.seed,
        "proposal_max_length": args.proposal_max_length,
        "neighbor_top_k": args.neighbor_top_k,
        "candidate_neighbors": args.candidate_neighbors,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(
                f"Neighbor cache {name}={config.get(name)!r}, expected {value!r}"
            )
    if int(config.get("num_neighbors", 0)) < args.num_neighbors:
        raise ValueError("Neighbor cache contains fewer neighbors than requested")
    rows = read_jsonl(path)
    cache = {(str(row["membership"]), int(row["source_index"])): row for row in rows}
    resolved = {}
    for membership, sample_index, sample in samples:
        key = (membership, int(sample["source_index"]))
        row = cache.get(key)
        if row is None:
            raise ValueError(f"Neighbor cache is missing {key}")
        if row.get("text") != sample["text"]:
            raise ValueError(f"Neighbor cache text mismatch for {key}")
        neighbors = row.get("neighbors")
        if not isinstance(neighbors, list) or len(neighbors) < args.num_neighbors:
            raise ValueError(
                f"Neighbor cache has fewer than {args.num_neighbors} neighbors for {key}"
            )
        resolved[(membership, sample_index)] = [
            str(value) for value in neighbors[: args.num_neighbors]
        ]
    return resolved


@torch.inference_mode()
def generate_neighbors(text, tokenizer, model, device, args):
    ids = tokenizer(
        text, truncation=True, max_length=args.proposal_max_length, return_tensors="pt"
    ).input_ids.to(device)
    replacements = {}
    dropout = torch.nn.Dropout(p=0.7)
    backbone = getattr(model, model.base_model_prefix)
    for position in range(1, ids.shape[1]):
        original = ids[0, position]
        embeds = backbone.embeddings(ids)
        embeds = torch.cat(
            (
                embeds[:, :position],
                dropout(embeds[:, position]).unsqueeze(1),
                embeds[:, position + 1 :],
            ),
            dim=1,
        )
        probabilities = torch.softmax(model(inputs_embeds=embeds).logits, dim=2)
        original_probability = probabilities[0, position, original].item()
        values, candidates = torch.topk(
            probabilities[:, position, :], args.neighbor_top_k, dim=1
        )
        denominator = 0.1 if original_probability == 1 else 1 - original_probability
        for candidate, value in zip(candidates[0], values[0]):
            if candidate != original:
                replacements[(position, int(candidate))] = value.item() / denominator
    result = []
    for position, candidate in nlargest(
        args.candidate_neighbors, replacements, key=replacements.get
    ):
        alternate = ids.clone()
        alternate[0, position] = candidate
        decoded = tokenizer.batch_decode(alternate)[0]
        decoded = decoded.replace(" [SEP]", " ").replace("[CLS] ", " ").strip()
        if decoded and decoded not in result:
            result.append(decoded)
    return result[: args.num_neighbors]


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def score_texts(wrapper, texts, max_length, batch_size):
    scores = []
    for batch in chunks(texts, batch_size):
        encoded = wrapper.tokenizer(
            list(batch), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded = {name: value.to(wrapper.device) for name, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention = encoded.get("attention_mask", torch.ones_like(input_ids))
        with wrapper.applied() as model:
            logits = model(**encoded, use_cache=False, return_dict=True).logits.float()
        if logits.shape[:2] != input_ids.shape and logits.shape[1] == input_ids.shape[0]:
            logits = logits.transpose(0, 1)
        labels = input_ids[:, 1:]
        valid = attention[:, 1:].bool() & attention[:, :-1].bool()
        for special_id in wrapper.tokenizer.all_special_ids:
            valid &= labels != special_id
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        counts = valid.sum(1)
        if (counts == 0).any():
            raise ValueError("No valid target tokens remain")
        scores.extend((-(losses * valid).sum(1) / counts).cpu().tolist())
    return scores


@torch.inference_mode()
def sequence_log_probabilities(wrapper, texts, max_length, batch_size):
    scores = []
    for batch in chunks(texts, batch_size):
        encoded = wrapper.tokenizer(
            list(batch), padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(wrapper.device) for name, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention = encoded.get("attention_mask", torch.ones_like(input_ids))
        with wrapper.applied() as model:
            logits = model(**encoded, use_cache=False, return_dict=True).logits.float()
        if logits.shape[:2] != input_ids.shape and logits.shape[1] == input_ids.shape[0]:
            logits = logits.transpose(0, 1)
        labels = input_ids[:, 1:]
        valid = attention[:, 1:].bool() & attention[:, :-1].bool()
        for special_id in wrapper.tokenizer.all_special_ids:
            valid &= labels != special_id
        token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        selected = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        scores.extend((selected * valid).sum(1).cpu().tolist())
    return scores


def evaluate_utility(wrapper, records, max_length, batch_size):
    from subj_steered_model.steering import chat_text

    texts = []
    for row in records:
        texts.extend(
            [
                chat_text(row["question"], row["answer_matching_behavior"]),
                chat_text(row["question"], row["answer_not_matching_behavior"]),
            ]
        )
    values = sequence_log_probabilities(wrapper, texts, max_length, batch_size)
    correct = sum(values[index] >= values[index + 1] for index in range(0, len(values), 2))
    return {"correct": int(correct), "total": len(records), "accuracy": correct / len(records)}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This experiment requires a GPU")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    interface_dir = Path(args.interface_dir).resolve()
    members, nonmembers, construction = load_samples(
        interface_dir,
        args.member_count,
        args.nonmember_count,
        member_start=args.member_start,
        nonmember_start=args.nonmember_start,
        construction_samples=args.construction_samples,
        construction_seed=args.construction_seed,
    )
    config = vars(args).copy()
    config.update(
        device=str(device),
        construction_pool=len(construction),
        construction_used=args.construction_samples,
    )
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(interface_dir))
    from subj_steered_model import create_subj_models

    construction_samples = args.construction_limit or args.construction_samples
    models = create_subj_models(
        args.model,
        dataset_path=interface_dir / "data/generate_dataset.json",
        device=device,
        dtype="bfloat16",
        layers=args.layers,
        read_token_index=args.read_token_index,
        clip=args.clip,
        noise_multiplier=args.noise_multiplier,
        multiplier=args.steering_multiplier,
        private_seed=args.private_seed,
        construction_samples=construction_samples,
        construction_seed=args.construction_seed,
        show_progress=True,
    )
    wrappers = {"mean": models.mean, "private": models.private}
    selected_names = ("mean", "private") if args.variants == "both" else (args.variants,)
    wrappers = {name: wrappers[name] for name in selected_names}

    checkpoint_paths = {name: output / name / "scores.csv" for name in selected_names}
    completed = {
        name: load_checkpoint(path) if args.resume else {} for name, path in checkpoint_paths.items()
    }
    failures = []
    started = time.monotonic()
    samples = [("member", i, row) for i, row in enumerate(members)] + [
        ("nonmember", i, row) for i, row in enumerate(nonmembers)
    ]
    cached_neighbors = None
    proposal_tokenizer = proposal_model = None
    if args.neighbor_cache:
        cached_neighbors = load_cached_neighbors(args.neighbor_cache, samples, args)
    else:
        proposal_tokenizer = AutoTokenizer.from_pretrained(args.proposal_model, use_fast=False)
        proposal_model = AutoModelForMaskedLM.from_pretrained(args.proposal_model).to(device).eval()
    for membership, sample_index, row in tqdm(samples, desc="Steering MIA"):
        key = (membership, sample_index)
        if all(key in completed[name] for name in completed):
            continue
        last_error = None
        for attempt in range(2):
            try:
                if cached_neighbors is None:
                    neighbors = generate_neighbors(
                        row["text"], proposal_tokenizer, proposal_model, device, args
                    )
                else:
                    neighbors = cached_neighbors[(membership, sample_index)]
                if not neighbors:
                    raise ValueError("No neighbors generated")
                texts = [row["text"], *neighbors]
                for name, wrapper in wrappers.items():
                    values = score_texts(
                        wrapper, texts, args.max_length, args.score_batch_size
                    )
                    ensure_finite(values, f"{name}/{membership}/{sample_index}")
                    completed[name][key] = {
                        "membership": membership,
                        "sample_index": sample_index,
                        "source_index": row["source_index"],
                        "label": row["label"],
                        "text": row["text"],
                        "original_score": values[0],
                        "neighbor_mean": float(np.mean(values[1:])),
                        "num_neighbors": len(values) - 1,
                        "score": values[0] - float(np.mean(values[1:])),
                    }
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                torch.cuda.empty_cache()
        if last_error is not None:
            failures.append(
                {"membership": membership, "sample_index": sample_index, "error": last_error}
            )
        done = sum(len(records) for records in completed.values())
        if done % (len(selected_names) * args.checkpoint_every) == 0 or last_error is not None:
            for name, records in completed.items():
                save_checkpoint(records.values(), checkpoint_paths[name])
            (output / "failures.json").write_text(
                json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    for name, records in completed.items():
        save_checkpoint(records.values(), checkpoint_paths[name])
    if failures:
        raise RuntimeError(f"{len(failures)} samples failed after retry; evaluation not run")
    summaries = {}
    for name, path in checkpoint_paths.items():
        summaries[name] = evaluate_scores(
            path, output / name / "evaluation", args.member_count, args.nonmember_count
        )
    utility = {}
    if args.utility_count:
        test_rows = read_json(interface_dir / "data/test_dataset_ab.json")
        utility_stop = args.utility_start + args.utility_count
        if args.utility_start < 0 or utility_stop > len(test_rows):
            raise ValueError("Requested utility slice is outside the SUBJ test pool")
        utility_rows = test_rows[args.utility_start:utility_stop]
        for name, wrapper in wrappers.items():
            utility[name] = evaluate_utility(
                wrapper, utility_rows, args.max_length, args.utility_batch_size
            )
            utility_path = output / name / "utility.json"
            utility_path.parent.mkdir(parents=True, exist_ok=True)
            utility_path.write_text(
                json.dumps(utility[name], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.monotonic() - started,
                "results": summaries,
                "utility": utility,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: value["auc"] for name, value in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
