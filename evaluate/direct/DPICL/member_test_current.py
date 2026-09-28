#!/usr/bin/env python3
"""Direct extraction/member test for samples produced by the DPICL runner.

The input is the JSON written by run_exp_sst2_local.py.  We deliberately use
``synthetic_text`` (or ``generated_text`` for WikiText-style records) as the
attack candidate and keep ``original_text`` as metadata only.
"""

import argparse
import csv
import difflib
import html
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

_DATASET_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_datasets"))
os.makedirs(_DATASET_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", _DATASET_CACHE)


def normalize_basic(text):
    text = html.unescape(str(text or "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_loose(text):
    text = normalize_basic(text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def best_similarity(candidate, reference):
    candidate = normalize_loose(candidate)
    reference = normalize_loose(reference)
    if not candidate or not reference:
        return 0.0
    scores = [difflib.SequenceMatcher(None, candidate, reference).ratio()]
    if len(candidate) >= len(reference):
        scores.append(difflib.SequenceMatcher(None, candidate[: len(reference)], reference).ratio())
    if len(reference) >= len(candidate):
        scores.append(difflib.SequenceMatcher(None, candidate, reference[: len(candidate)]).ratio())
    return max(scores)


def best_window_similarity(candidate, reference):
    """Compare a continuation with local windows inside a WikiText paragraph.

    ``SequenceMatcher`` is run once on token sequences instead of once for
    every possible character window.  The old implementation was quadratic
    in the number of reference windows and made 100-candidate attacks appear
    stuck.
    """
    candidate_tokens = normalize_loose(candidate).split()
    reference_tokens = normalize_loose(reference).split()
    return best_window_similarity_tokens(candidate_tokens, reference_tokens)


def best_window_similarity_tokens(candidate_tokens, reference_tokens):
    """Return the best local token-window similarity without rescanning text."""
    if not candidate_tokens or not reference_tokens:
        return 0.0
    if len(candidate_tokens) > len(reference_tokens):
        return difflib.SequenceMatcher(
            None, candidate_tokens, reference_tokens, autojunk=False
        ).ratio()

    window_size = len(candidate_tokens)
    matcher = difflib.SequenceMatcher(
        None, candidate_tokens, reference_tokens, autojunk=False
    )
    best = 0.0
    # Evaluate only windows centered around matching blocks.  This preserves
    # local fuzzy behavior while avoiding a full sliding-window scan.
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        start = max(0, min(block.b, len(reference_tokens) - window_size))
        window = reference_tokens[start : start + window_size]
        score = difflib.SequenceMatcher(
            None, candidate_tokens, window, autojunk=False
        ).ratio()
        best = max(best, score)
        if best >= 1.0:
            break
    return best


def load_generated_candidates(path, field):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Expected a list or a results list in {path}")

    candidates = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            text = record
            metadata = {}
        elif isinstance(record, dict):
            text = record.get(field)
            if text is None and field == "synthetic_text":
                text = record.get("generated_text")
            if text is None:
                text = record.get("text")
            metadata = record
        else:
            continue
        # Keep missing/empty/short records as explicit unmatched candidates so
        # the output count remains identical to the input JSON count.
        candidates.append({"candidate_id": index, "generated": str(text or "").strip(), "metadata": metadata})
    return candidates


def load_text_records(path, task):
    """Load SST-2 sentences or WikiText paragraphs from local datasets."""
    if os.path.isdir(path):
        try:
            from datasets import load_dataset, load_from_disk

            if task == "wikitext" and not os.path.exists(os.path.join(path, "dataset_dict.json")):
                files = {
                    split: os.path.join(path, f"{split}-00000-of-00001.parquet")
                    for split in ("train", "validation", "test")
                }
                missing = [file for file in files.values() if not os.path.exists(file)]
                if missing:
                    raise FileNotFoundError("Missing WikiText files: " + ", ".join(missing))
                dataset = load_dataset("parquet", data_files=files)
            else:
                dataset = load_from_disk(path)
            if hasattr(dataset, "keys") and "train" in dataset:
                dataset = dataset["train"]
            records = []
            for row in dataset:
                text_key = "text" if task == "wikitext" else "sentence"
                text = str(row.get(text_key, "")).strip()
                if text:
                    records.append({"idx": len(records), "text": text, "label": int(row.get("label", -1)) if task == "sst2" else None})
            return records
        except Exception as exc:
            raise RuntimeError(f"Could not load {task} dataset from {path}: {exc}") from exc

    if task != "sst2":
        raise ValueError("WikiText training data must be a local datasets directory or parquet directory")
    records = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            text = str(row.get("sentence", "")).strip()
            if text:
                records.append({"idx": len(records), "text": text, "label": int(row.get("label", -1))})
    return records


def candidate_records(records, top_k, token_index):
    loose = normalize_loose(records["generated"])
    tokens = set(re.findall(r"[a-z0-9]{4,}", loose))
    if not tokens:
        return list(range(min(top_k, len(token_index["records"]))))
    counts = Counter()
    for token in tokens:
        for index in token_index["tokens"].get(token, ()):
            counts[index] += 1
    if not counts:
        return list(range(min(top_k, len(token_index["records"]))))
    return [index for index, _ in counts.most_common(top_k)]


def run_member_test(args):
    generated = load_generated_candidates(args.generated_json, args.candidate_field)
    if not generated:
        raise ValueError("No valid generated candidates found in the input JSON")

    train = load_text_records(args.train_data, args.task)
    if not train:
        raise ValueError(f"The {args.task} training set is empty")
    print(f"Loaded {len(train)} {args.task} training records", flush=True)
    # Cache normalized forms once; recomputing them for every candidate and
    # every reference dominated runtime on WikiText.
    for record in train:
        record["_basic"] = normalize_basic(record["text"])
        record["_loose"] = normalize_loose(record["text"])
        record["_tokens"] = record["_loose"].split()
    token_index = {"records": train, "tokens": defaultdict(set)}
    for record in train:
        for token in set(re.findall(r"[a-z0-9]{4,}", normalize_loose(record["text"]))):
            token_index["tokens"][token].add(record["idx"])

    results = []
    candidate_limit = args.top_k_candidates
    if args.task == "wikitext":
        candidate_limit = min(candidate_limit, args.wikitext_max_candidates)
    for candidate_position, candidate in enumerate(generated, start=1):
        if candidate_position == 1 or candidate_position % 10 == 0 or candidate_position == len(generated):
            print(f"Processing {candidate_position}/{len(generated)} candidates", flush=True)
        generated_text = candidate["generated"]
        raw = generated_text.strip()
        basic = normalize_basic(raw)
        loose = normalize_loose(raw)

        if len(loose) < args.min_chars:
            results.append({
                "candidate_id": candidate["candidate_id"],
                "generated": raw,
                "match_type": "unmatched",
                "reason": "below_min_chars",
                "score": 0.0,
                "metadata": candidate["metadata"],
            })
            continue

        best = None
        best_score = 0.0
        match_type = "unmatched"
        for index in candidate_records(candidate, candidate_limit, token_index):
            reference = train[index]
            reference_basic = reference["_basic"]
            reference_loose = reference["_loose"]
            if raw == reference["text"].strip():
                best, best_score, match_type = reference, 1.0, "raw_exact"
                break
            if basic == reference_basic:
                best, best_score, match_type = reference, 1.0, "normalized_exact"
                break
            if len(loose) >= args.contains_min_chars and (
                loose in reference_loose or reference_loose in loose
            ):
                best, best_score, match_type = reference, 1.0, "contains"
                break
            if args.task == "wikitext":
                candidate_tokens = normalize_loose(raw).split()
                score = (
                    best_window_similarity_tokens(candidate_tokens, reference["_tokens"])
                    if len(candidate_tokens) >= args.window_min_tokens
                    else 0.0
                )
                candidate_match_type = "window_fuzzy"
            else:
                score = best_similarity(raw, reference["text"])
                candidate_match_type = "fuzzy"
            if score > best_score:
                best, best_score, match_type = reference, score, candidate_match_type

        accepted = match_type in {"raw_exact", "normalized_exact", "contains"} or (
            best_score >= args.fuzzy_threshold
        )
        result = {
            "candidate_id": candidate["candidate_id"],
            "generated": raw,
            "match_type": match_type if accepted else "unmatched",
            "reason": "" if accepted else "no_match_or_below_fuzzy_threshold",
            "score": float(best_score),
            "metadata": candidate["metadata"],
        }
        if best is not None and (result["match_type"] != "unmatched"):
            result.update({"train_idx": best["idx"], "gold": best["text"]})
            if args.task == "sst2":
                result["label"] = best["label"]
        results.append(result)

    matched = [item for item in results if item["match_type"] != "unmatched"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"member_test_{timestamp}"
    json_path = os.path.join(output_dir, stem + ".json")
    csv_path = os.path.join(output_dir, stem + ".csv")
    summary_path = os.path.join(output_dir, stem + "_summary.txt")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "match_type", "reason", "score", "train_idx", "label", "generated", "gold"])
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"task: {args.task}\n")
        handle.write(f"generated candidates: {len(results)}\n")
        handle.write(f"matched candidates: {len(matched)}\n")
        handle.write(f"extraction/match rate: {len(matched) / len(results):.4%}\n")
        handle.write(f"fuzzy threshold: {args.fuzzy_threshold}\n")
        for match_type, count in sorted(Counter(item["match_type"] for item in results).items()):
            handle.write(f"{match_type}: {count}\n")
    print(f"Candidates: {len(results)}")
    print(f"Matched: {len(matched)} ({len(matched) / len(results):.2%})")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


def parse_args():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Direct member test for DPICL generated samples")
    parser.add_argument("--task", choices=("sst2", "wikitext"), default="sst2")
    parser.add_argument("--generated-json", default=None)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--candidate-field", default=None)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    parser.add_argument("--contains-min-chars", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument(
        "--window-min-tokens",
        type=int,
        default=8,
        help="Minimum WikiText continuation length for window fuzzy matching.",
    )
    parser.add_argument("--top-k-candidates", type=int, default=1000)
    parser.add_argument(
        "--wikitext-max-candidates",
        type=int,
        default=300,
        help="Maximum reference paragraphs scored per WikiText candidate.",
    )
    args = parser.parse_args()
    if args.task == "wikitext":
        default_json = os.path.join(root, "results", "chatglm-wikitext", "synthetic_text_results.json")
        default_train = os.path.join(root, "..", "data", "wikitext_2_raw_v1")
        default_field = "generated_text"
    else:
        default_json = os.path.join(root, "results", "chatglm-sst2", "synthetic_text_results.json")
        default_train = os.path.join(root, "..", "data", "sst2")
        default_field = "synthetic_text"
    args.generated_json = args.generated_json or default_json
    args.train_data = args.train_data or default_train
    args.candidate_field = args.candidate_field or default_field
    return args


if __name__ == "__main__":
    run_member_test(parse_args())
