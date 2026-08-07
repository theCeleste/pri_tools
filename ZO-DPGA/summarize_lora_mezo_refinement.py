import argparse
import json
from pathlib import Path

import yaml

from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml
from run_lora_mezo_refinement import CONFIG_KEYS, DEFAULT_CONFIG, DEFAULT_SELECTION
from summarize_lora_mezo_exploration import confirm_groups, load_rows, write_csv


EXPECTED = {"boundary": 12, "architecture": 7, "scale": 7, "horizon": 6}


def update_selection(path, stage, rows):
    if len(rows) != EXPECTED[stage]:
        raise ValueError(f"Refusing incomplete {stage}: expected {EXPECTED[stage]}, found {len(rows)}")
    selection = load_yaml(path)
    best = rows[0]
    if stage == "boundary":
        selection["boundary_best"] = {key: best[key] for key in ("learning_rate", "zo_eps")}
    elif stage == "architecture":
        selection["architecture_best"] = {key: best[key] for key in (
            "learning_rate", "zo_eps", "lora_target_modules", "lora_num_layers")}
    elif stage == "scale":
        selection["scale_best"] = {key: best[key] for key in (
            "learning_rate", "zo_eps", "lora_target_modules", "lora_num_layers",
            "lora_r", "lora_alpha")}
    else:
        selection["finalists"] = [{key: row[key] for key in CONFIG_KEYS} for row in rows[:2]]
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")
    print(f"Updated {path} from {stage}")


def main():
    parser = argparse.ArgumentParser(description="Summarize LoRA + MeZO refinement")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=("boundary", "architecture", "scale", "horizon", "confirm"), required=True)
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
        write_csv(output / "groups.csv", groups)
        (output / "groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    elif args.update_selection:
        update_selection(args.selection, args.stage, rows)


if __name__ == "__main__":
    main()
