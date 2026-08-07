import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from run_formal_experiment import command_for


LLAMA_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_zo_dpga_matrix.yaml"
DEFAULT_SELECTION = LLAMA_DIR / "configs" / "lora_zo_dpga_selection.yaml"
REQUIRED_TUNING_KEYS = (
    "learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps",
    "batch_size", "max_steps",
)


def load_yaml(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"Unsupported or invalid version in {path}")
    return data


def require_values(mapping, keys, label):
    missing = [key for key in keys if mapping.get(key) is None]
    if missing:
        raise ValueError(f"{label} is incomplete; set {', '.join(missing)} in the selection file")


def validate_stage1_selection(config, selection):
    selected = dict(selection.get("stage1_best") or {})
    require_values(selected, ("learning_rate", "dp_clip"), "stage1_best")
    grid = config["stages"]["stage1"]["grid"]
    if selected["learning_rate"] not in grid["learning_rate"]:
        raise ValueError("stage1_best.learning_rate is not a member of the stage1 grid")
    if selected["dp_clip"] not in grid["dp_clip"]:
        raise ValueError("stage1_best.dp_clip is not a member of the stage1 grid")
    return selected


def validate_stage2_selection(config, selection):
    stage1_best = validate_stage1_selection(config, selection)
    selected = dict(selection.get("stage2_best") or {})
    require_values(
        selected,
        ("learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps"),
        "stage2_best",
    )
    if any(selected[key] != stage1_best[key] for key in stage1_best):
        raise ValueError("stage2_best must retain the selected stage1 learning_rate and dp_clip")
    stage = config["stages"]["stage2"]
    pair = {"lora_r": selected["lora_r"], "lora_alpha": selected["lora_alpha"]}
    if pair not in stage["lora_pairs"]:
        raise ValueError("stage2_best LoRA rank/alpha is not a member of the stage2 grid")
    if selected["zo_eps"] not in stage["zo_eps_values"]:
        raise ValueError("stage2_best.zo_eps is not a member of the stage2 grid")
    return selected


def validate_finalists(config, selection):
    stage2_best = validate_stage2_selection(config, selection)
    finalists = selection.get("finalists") or []
    if len(finalists) != 2:
        raise ValueError("finalists must contain exactly two configurations")
    budgets = config["stages"]["stage3"]["budgets"]
    normalized = []
    for index, candidate in enumerate(finalists):
        candidate = dict(candidate)
        require_values(candidate, REQUIRED_TUNING_KEYS, f"finalists[{index}]")
        inherited_keys = ("learning_rate", "dp_clip", "lora_r", "lora_alpha", "zo_eps")
        if any(candidate[key] != stage2_best[key] for key in inherited_keys):
            raise ValueError(f"finalists[{index}] must retain all stage2_best values")
        budget = {"batch_size": candidate["batch_size"], "max_steps": candidate["max_steps"]}
        if budget not in budgets:
            raise ValueError(f"finalists[{index}] batch_size/max_steps is not a stage3 budget")
        normalized.append(candidate)
    if normalized[0] == normalized[1]:
        raise ValueError("The two finalists must be different configurations")
    return normalized


def stage_parameter_sets(config, selection, stage_name):
    stage = config["stages"][stage_name]
    fixed = dict(stage.get("fixed", {}))
    if stage_name == "stage1":
        grid = stage["grid"]
        return [
            fixed | {"learning_rate": learning_rate, "dp_clip": dp_clip}
            for learning_rate, dp_clip in itertools.product(
                grid["learning_rate"], grid["dp_clip"]
            )
        ]
    if stage_name == "stage2":
        inherited = validate_stage1_selection(config, selection)
        return [
            fixed | inherited | dict(pair) | {"zo_eps": zo_eps}
            for pair, zo_eps in itertools.product(stage["lora_pairs"], stage["zo_eps_values"])
        ]
    if stage_name == "stage3":
        inherited = validate_stage2_selection(config, selection)
        return [fixed | inherited | dict(budget) for budget in stage["budgets"]]
    if stage_name == "final":
        return [
            dict(candidate) | {"seed": seed}
            for candidate, seed in itertools.product(
                validate_finalists(config, selection), stage["seeds"]
            )
        ]
    raise ValueError(f"Unknown stage: {stage_name}")


def value_tag(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def expand_stage(config, selection, stage_name):
    common_base = dict(config["common"])
    jobs = []
    for ordinal, parameters in enumerate(stage_parameter_sets(config, selection, stage_name)):
        common = common_base | parameters
        missing = [key for key in REQUIRED_TUNING_KEYS if common.get(key) is None]
        if missing:
            raise ValueError(f"{stage_name} job is missing parameters: {missing}")
        experiment_id = (
            f"lora-zo_dpga-{stage_name}-{ordinal:02d}"
            f"-lr{value_tag(common['learning_rate'])}"
            f"-clip{value_tag(common['dp_clip'])}"
            f"-r{common['lora_r']}-a{common['lora_alpha']}"
            f"-mu{value_tag(common['zo_eps'])}"
            f"-bs{common['batch_size']}-steps{common['max_steps']}"
            f"-seed{common['seed']}"
        )
        output_dir = Path(config["output_root"]) / stage_name / experiment_id
        jobs.append({
            "suite": f"lora_zo_dpga_{stage_name}",
            "experiment_id": experiment_id,
            "description": config["stages"][stage_name]["description"],
            "model_name": config["model_name"],
            "method": "zo_dpga",
            "mode": "lora",
            "seed": common["seed"],
            "dp_epsilon": common["dp_epsilon"],
            "dev_only": bool(common["dev_only"]),
            "output_dir": output_dir,
            "common": common,
            "mode_config": {
                "lora_r": common["lora_r"],
                "lora_alpha": common["lora_alpha"],
            },
        })
    return jobs


def snapshot_for(job, command, config_path, selection_path):
    snapshot = dict(job)
    snapshot["output_dir"] = str(job["output_dir"])
    snapshot["command"] = command
    snapshot["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    snapshot["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    return snapshot


def execute_job(job, config_path, selection_path, resume=False):
    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output exists: {output_dir}. Use --resume only for an intended resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(job)
    snapshot = snapshot_for(job, command, config_path, selection_path)
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != command:
            raise ValueError("Existing run_config.json does not match the requested resume command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=LLAMA_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the staged LoRA + ZODPGA parameter matrix")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "final"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run", action="store_true", help="Execute exactly one selected job")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    selection = load_yaml(args.selection)
    jobs = expand_stage(config, selection, args.stage)
    if args.index is not None and not 0 <= args.index < len(jobs):
        raise IndexError(f"index must be between 0 and {len(jobs) - 1}")
    selected = jobs if args.index is None else [jobs[args.index]]

    print(f"Stage {args.stage}: {len(jobs)} jobs")
    for index, job in enumerate(jobs):
        marker = "*" if job in selected else " "
        print(f"{marker} [{index:02d}] {job['experiment_id']} -> {job['output_dir']}")
    if not args.run:
        if args.index is not None:
            print("Command:", shlex.join(command_for(selected[0])))
        print("Dry run only. Add --index N --run to execute exactly one job.")
        return
    if args.index is None:
        raise ValueError("Execution requires --index N; use the stage runner for sequential execution")
    execute_job(selected[0], args.config, args.selection, resume=args.resume)


if __name__ == "__main__":
    main()
