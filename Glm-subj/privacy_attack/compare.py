#!/usr/bin/env python3
"""Combine direct extraction and indirect MIA results into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import write_json


FIELDS = [
    "method", "threat_model", "targets", "generations",
    "target_exact_hits", "target_normalized_exact_hits", "target_fuzzy_hits",
    "generation_exact_hits", "generation_normalized_exact_hits", "generation_fuzzy_hits",
    "exact_success_rate", "normalized_exact_success_rate", "fuzzy_success_rate", "mean_fuzzy_score",
    "mean_rouge_l", "mean_token_f1", "mean_char_similarity",
    "mean_prefix_match_tokens", "mean_continuous_token_match",
    "target_hits_at_4_tokens", "eligible_targets_at_4_tokens", "hit_rate_at_4_tokens",
    "target_hits_at_8_tokens", "eligible_targets_at_8_tokens", "hit_rate_at_8_tokens",
    "fuzzy_train_minus_holdout", "mia_best_metric", "mia_auc",
    "mia_tpr_at_1pct_fpr", "target_epsilon", "actual_epsilon", "delta",
]


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--result", action="append", nargs="+", required=True, metavar="VALUE",
        help="METHOD THREAT PREFIX_SUMMARY [MIA_SUMMARY]",
    )
    value.add_argument(
        "--require-matched-targets", action="store_true",
        help="Require every prefix result to contain the same attacked prefixes and references",
    )
    value.add_argument("--output-json", default="outputs/attack_comparison.json")
    value.add_argument("--output-csv", default="outputs/attack_comparison.csv")
    return value


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def privacy_for(prefix):
    adapter = prefix.get("configuration", {}).get("adapter")
    report_path = Path(adapter) / "training_report.json" if adapter else None
    if not report_path or not report_path.is_file():
        return {}
    return load_json(report_path).get("privacy") or {}


def load_attack_targets(summary_path):
    records_path = Path(summary_path).with_name("records.jsonl")
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    targets = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["split"], int(record["dataset_index"]), int(record["sample_index"]))
            targets[key] = (record["prefix"], record["reference"])
    return targets


def direct_attack_configuration(summary_path):
    configuration = load_json(summary_path).get("configuration", {})
    keys = (
        "targets_per_split", "samples_per_prefix", "prefix_tokens", "min_suffix_tokens",
        "max_new_tokens", "temperature_start", "temperature_end", "top_p", "top_k",
        "fuzzy_threshold", "minimum_fuzzy_tokens", "seed", "prompt_mode",
    )
    return {key: configuration.get(key) for key in keys}


def main(argv=None):
    args = parser().parse_args(argv)
    invalid = [result for result in args.result if len(result) not in (3, 4)]
    if invalid:
        raise ValueError("Each --result needs METHOD THREAT PREFIX_SUMMARY [MIA_SUMMARY]")
    if args.require_matched_targets:
        baseline = load_attack_targets(args.result[0][2])
        baseline_configuration = direct_attack_configuration(args.result[0][2])
        for result in args.result[1:]:
            if load_attack_targets(result[2]) != baseline:
                raise ValueError(
                    f"Direct-attack targets do not match between {args.result[0][0]!r} "
                    f"and {result[0]!r}"
                )
            if direct_attack_configuration(result[2]) != baseline_configuration:
                raise ValueError(
                    f"Direct-attack sampling configuration does not match between "
                    f"{args.result[0][0]!r} and {result[0]!r}"
                )

    rows = []
    for result in args.result:
        method, threat, prefix_path = result[:3]
        mia_path = result[3] if len(result) == 4 else None
        prefix = load_json(prefix_path)
        mia = load_json(mia_path) if mia_path else None
        train = prefix["train"]
        difference = prefix["train_minus_holdout"]
        best = mia["best"] if mia else {}
        privacy = privacy_for(prefix)
        rows.append({
            "method": method, "threat_model": threat,
            "targets": train["targets"], "generations": train.get("generations"),
            "target_exact_hits": train.get(
                "target_exact_hits", round(train["exact_success_rate"] * train["targets"])
            ),
            "target_normalized_exact_hits": train.get(
                "target_normalized_exact_hits",
                round(train["normalized_exact_success_rate"] * train["targets"]),
            ),
            "target_fuzzy_hits": train.get(
                "target_fuzzy_hits", round(train["fuzzy_success_rate"] * train["targets"])
            ),
            "generation_exact_hits": train.get("generation_exact_hits"),
            "generation_normalized_exact_hits": train.get("generation_normalized_exact_hits"),
            "generation_fuzzy_hits": train.get("generation_fuzzy_hits"),
            "exact_success_rate": train["exact_success_rate"],
            "normalized_exact_success_rate": train["normalized_exact_success_rate"],
            "fuzzy_success_rate": train["fuzzy_success_rate"],
            "mean_fuzzy_score": train["mean_fuzzy_score"], "mean_rouge_l": train["mean_rouge_l"],
            "mean_token_f1": train["mean_token_f1"], "mean_char_similarity": train["mean_char_similarity"],
            "mean_prefix_match_tokens": train["mean_prefix_match_tokens"],
            "mean_continuous_token_match": train["mean_continuous_token_match"],
            "target_hits_at_4_tokens": train.get(
                "target_hits_at_4_tokens", round(train["hit_rate_at_4_tokens"] * train["targets"])
            ),
            "eligible_targets_at_4_tokens": train.get("eligible_targets_at_4_tokens", train["targets"]),
            "hit_rate_at_4_tokens": train["hit_rate_at_4_tokens"],
            "target_hits_at_8_tokens": train.get(
                "target_hits_at_8_tokens", round(train["hit_rate_at_8_tokens"] * train["targets"])
            ),
            "eligible_targets_at_8_tokens": train.get("eligible_targets_at_8_tokens", train["targets"]),
            "hit_rate_at_8_tokens": train["hit_rate_at_8_tokens"],
            "fuzzy_train_minus_holdout": difference["fuzzy_success_rate"],
            "mia_best_metric": mia["best_metric"] if mia else None,
            "mia_auc": best.get("auc"),
            "mia_tpr_at_1pct_fpr": best.get("tpr_at_1pct_fpr"),
            "target_epsilon": privacy.get("target_epsilon"),
            "actual_epsilon": privacy.get("actual_epsilon"), "delta": privacy.get("delta"),
        })
    pairwise = None
    if len(rows) == 2:
        difference_keys = (
            "target_exact_hits", "target_normalized_exact_hits", "target_fuzzy_hits",
            "generation_exact_hits", "generation_normalized_exact_hits", "generation_fuzzy_hits",
            "exact_success_rate", "normalized_exact_success_rate", "fuzzy_success_rate",
            "mean_fuzzy_score", "mean_rouge_l", "mean_token_f1", "mean_char_similarity",
            "mean_prefix_match_tokens", "mean_continuous_token_match",
            "target_hits_at_4_tokens", "hit_rate_at_4_tokens",
            "target_hits_at_8_tokens", "hit_rate_at_8_tokens",
        )
        pairwise = {
            "operation": "second_minus_first",
            "first": rows[0]["method"],
            "second": rows[1]["method"],
            "differences": {
                key: rows[1][key] - rows[0][key]
                for key in difference_keys
                if rows[0].get(key) is not None and rows[1].get(key) is not None
            },
        }

    payload = {
        "comparison_unit": "same ChatGLM3 base and Subj source data; compare matched threat models",
        "matched_direct_attack_targets": args.require_matched_targets,
        "warning": (
            "Known protected text, source-original recovery, construction membership, and direct "
            "training membership are different threat models. Source-original causal MIA is a proxy only."
        ),
        "rows": rows,
        "pairwise_comparison": pairwise,
    }
    write_json(args.output_json, payload)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
