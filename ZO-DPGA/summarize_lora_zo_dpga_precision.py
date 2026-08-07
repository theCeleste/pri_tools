import argparse
import json
from pathlib import Path

import yaml

from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml
from run_lora_zo_dpga_precision import CONFIG_KEYS, DEFAULT_CONFIG, DEFAULT_SELECTION
from summarize_lora_zo_dpga_exploration import confirm_groups, load_rows, write_csv


EXPECTED = {"core": 12, "privacy_update": 9, "rank": 3, "batch": 4}
REFERENCE_DPZERO_MEAN = 0.828


def update_selection(path, stage, rows):
    if len(rows) != EXPECTED[stage]:
        raise ValueError(f"Refusing incomplete {stage}: expected {EXPECTED[stage]}, found {len(rows)}")
    selection = load_yaml(path)
    best = rows[0]
    if stage == "core":
        selection["core_best"] = {
            key: best[key] for key in ("zo_eps", "lora_num_layers", "max_steps")
        }
    elif stage == "privacy_update":
        selection["privacy_best"] = {
            key: best[key] for key in (
                "learning_rate", "dp_clip", "zo_eps", "lora_num_layers", "max_steps"
            )
        }
    elif stage == "rank":
        selection["rank_best"] = {
            key: best[key] for key in (
                "learning_rate", "dp_clip", "zo_eps", "lora_target_modules",
                "lora_num_layers", "lora_r", "lora_alpha", "max_steps"
            )
        }
    else:
        selection["finalists"] = [
            {key: row[key] for key in CONFIG_KEYS} for row in rows[:2]
        ]
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    print(f"Updated {path} from {stage}")


def main():
    parser = argparse.ArgumentParser(description="Summarize ZODPGA-only precision search")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=("core", "privacy_update", "rank", "batch", "confirm"), required=True)
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
    if args.stage == "confirm":
        groups = confirm_groups(rows)
        for group in groups:
            group.pop("beats_round1_mean", None)
            group["beats_zo_dpga_mean_0_828"] = bool(
                group["runs"] == 3 and group["mean_dev_accuracy"] > REFERENCE_DPZERO_MEAN
            )
        write_csv(output / "groups.csv", groups)
        (output / "groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    elif args.update_selection:
        update_selection(args.selection, args.stage, rows)


if __name__ == "__main__":
    main()
