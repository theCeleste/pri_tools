import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import yaml

from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml
from run_lora_mezo_exploration import (
    CONFIG_KEYS, DEFAULT_CONFIG, DEFAULT_SELECTION, validate_optimizer_selection,
    validate_structure_selection,
)


def trainer_details(output_dir):
    states = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        return None, None
    state = json.loads(states[-1].read_text(encoding="utf-8"))
    match = re.search(r"checkpoint-(\d+)", state.get("best_model_checkpoint") or "")
    best_step = int(match.group(1)) if match else None
    eval_loss = next(
        (row.get("eval_loss") for row in state.get("log_history", [])
         if row.get("step") == best_step and "eval_loss" in row),
        None,
    )
    return best_step, eval_loss


def load_rows(root, stage):
    rows = []
    for config_file in sorted((root / stage).glob("*/run_config.json")):
        metrics_file = config_file.parent / "eval_metrics.json"
        if not metrics_file.exists():
            continue
        run = json.loads(config_file.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metric_name = next((key for key in metrics if key.endswith("accuracy")), next(iter(metrics), None))
        best_step, eval_loss = trainer_details(config_file.parent)
        common = run["common"]
        rows.append({
            "experiment_id": run["experiment_id"], "seed": run["seed"],
            **{key: common[key] for key in CONFIG_KEYS},
            "metric_name": metric_name,
            "metric_value": metrics.get(metric_name) if metric_name else None,
            "best_checkpoint_step": best_step, "best_eval_loss": eval_loss,
            "output_dir": str(config_file.parent),
        })
    return sorted(rows, key=lambda row: row["metric_value"] if row["metric_value"] is not None else -1, reverse=True)


def update_selection(path, stage, rows):
    expected = {"optimizer": 12, "structure": 8, "budget": 6}
    if len(rows) != expected[stage]:
        raise ValueError(f"Refusing incomplete {stage}: expected {expected[stage]}, found {len(rows)}")
    selection = load_yaml(path)
    if stage == "optimizer":
        selection["optimizer_best"] = {key: rows[0][key] for key in ("learning_rate", "zo_eps")}
    elif stage == "structure":
        selection["structure_best"] = {
            key: rows[0][key] for key in (
                "learning_rate", "zo_eps", "lora_r", "lora_alpha",
                "lora_target_modules", "lora_num_layers",
            )
        }
    else:
        selection["finalists"] = [{key: row[key] for key in CONFIG_KEYS} for row in rows[:2]]
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    print(f"Updated {path} from {stage}")


def confirm_groups(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in CONFIG_KEYS)
        groups.setdefault(key, []).append(row)
    result = []
    for values, group in groups.items():
        metrics = [row["metric_value"] for row in group if row["metric_value"] is not None]
        result.append({
            **dict(zip(CONFIG_KEYS, values)), "runs": len(metrics),
            "mean_dev_accuracy": statistics.mean(metrics) if metrics else None,
            "seed_sd": statistics.stdev(metrics) if len(metrics) > 1 else None,
        })
    return sorted(result, key=lambda row: row["mean_dev_accuracy"] if row["mean_dev_accuracy"] is not None else -1, reverse=True)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize LoRA + MeZO exploration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=("optimizer", "structure", "budget", "confirm"), required=True)
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
    if rows: print(f"Best: {rows[0]['experiment_id']} metric={rows[0]['metric_value']}")
    if args.stage == "confirm":
        groups = confirm_groups(rows)
        write_csv(output / "groups.csv", groups)
        (output / "groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    elif args.update_selection:
        update_selection(args.selection, args.stage, rows)


if __name__ == "__main__":
    main()
