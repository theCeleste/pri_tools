import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import yaml

from run_lora_zo_dpga_matrix import DEFAULT_CONFIG, DEFAULT_SELECTION, LLAMA_DIR, load_yaml


def first_metric(metrics):
    preferred = [key for key in metrics if key.endswith("accuracy")]
    key = preferred[0] if preferred else next(iter(metrics), None)
    return key, metrics.get(key) if key else None


def trainer_details(output_dir):
    states = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        return None, None
    state = json.loads(states[-1].read_text(encoding="utf-8"))
    checkpoint = state.get("best_model_checkpoint") or ""
    match = re.search(r"checkpoint-(\d+)", checkpoint)
    best_step = int(match.group(1)) if match else None
    best_eval_loss = None
    if best_step is not None:
        for row in state.get("log_history", []):
            if row.get("step") == best_step and "eval_loss" in row:
                best_eval_loss = row["eval_loss"]
    return best_step, best_eval_loss


def load_rows(root, stage):
    rows = []
    for config_file in sorted((root / stage).glob("*/run_config.json")):
        output_dir = config_file.parent
        metrics_file = output_dir / "eval_metrics.json"
        if not metrics_file.exists():
            continue
        run = json.loads(config_file.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metric_name, metric_value = first_metric(metrics)
        best_step, eval_loss = trainer_details(output_dir)
        common = run["common"]
        rows.append({
            "experiment_id": run["experiment_id"],
            "seed": run["seed"],
            "learning_rate": common["learning_rate"],
            "dp_clip": common["dp_clip"],
            "lora_r": common["lora_r"],
            "lora_alpha": common["lora_alpha"],
            "zo_eps": common["zo_eps"],
            "batch_size": common["batch_size"],
            "max_steps": common["max_steps"],
            "metric_name": metric_name,
            "metric_value": metric_value,
            "best_checkpoint_step": best_step,
            "best_eval_loss": eval_loss,
            "output_dir": str(output_dir),
        })
    return sorted(rows, key=lambda row: (row["metric_value"] is not None, row["metric_value"]), reverse=True)


def tuning_values(row):
    return {
        key: row[key]
        for key in (
            "learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps",
            "batch_size", "max_steps",
        )
    }


def update_selection(path, stage, rows):
    expected_counts = {"stage1": 16, "stage2": 12, "stage3": 3}
    if stage in expected_counts and len(rows) != expected_counts[stage]:
        raise ValueError(
            f"Refusing to select from an incomplete {stage}: "
            f"expected {expected_counts[stage]} completed runs, found {len(rows)}"
        )
    if not rows or rows[0]["metric_value"] is None:
        raise ValueError(f"No completed metric is available for {stage}")
    selection = load_yaml(path)
    if stage == "stage1":
        selection["stage1_best"] = {
            "learning_rate": rows[0]["learning_rate"],
            "dp_clip": rows[0]["dp_clip"],
        }
    elif stage == "stage2":
        selection["stage2_best"] = {
            key: rows[0][key]
            for key in ("learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps")
        }
    elif stage == "stage3":
        if len(rows) < 2:
            raise ValueError("At least two completed stage3 runs are required to select finalists")
        selection["finalists"] = [tuning_values(row) for row in rows[:2]]
    else:
        raise ValueError("The final stage does not update the selection file")
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    print(f"Updated {path} from the ranked {stage} results")


def final_summary(rows):
    grouped = {}
    keys = (
        "learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps",
        "batch_size", "max_steps",
    )
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries = []
    for values, group in grouped.items():
        metrics = [row["metric_value"] for row in group if row["metric_value"] is not None]
        mean = statistics.mean(metrics) if metrics else None
        sd = statistics.stdev(metrics) if len(metrics) > 1 else None
        summaries.append({
            **dict(zip(keys, values)),
            "runs": len(metrics),
            "mean_dev_accuracy": mean,
            "seed_sd": sd,
            "seeds_above_0_634": sum(value > 0.634 for value in metrics),
            "passes_success_rule": bool(
                len(metrics) == 3 and mean is not None and mean >= 0.645
                and sum(value > 0.634 for value in metrics) >= 2
            ),
        })
    return sorted(summaries, key=lambda row: row["mean_dev_accuracy"] or -1, reverse=True)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Rank and summarize the LoRA + ZODPGA matrix")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "final"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-selection", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    root = LLAMA_DIR / config["output_root"]
    rows = load_rows(root, args.stage)
    output = args.output or root / "summary" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", rows)
    (output / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Ranked {len(rows)} completed {args.stage} runs at {output}")
    if rows:
        print(f"Best: {rows[0]['experiment_id']} metric={rows[0]['metric_value']}")
    if args.stage == "final":
        summary = final_summary(rows)
        write_csv(output / "groups.csv", summary)
        (output / "groups.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.update_selection:
        update_selection(args.selection, args.stage, rows)


if __name__ == "__main__":
    main()
