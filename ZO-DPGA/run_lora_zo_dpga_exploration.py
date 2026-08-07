import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml, require_values


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_zo_dpga_exploration.yaml"
DEFAULT_SELECTION = LLAMA_DIR / "configs" / "lora_zo_dpga_exploration_selection.yaml"
STAGES = ("boundary", "structure", "schedule", "confirm")
TUNING_KEYS = (
    "learning_rate", "dp_clip", "zo_eps", "lora_r", "lora_alpha",
    "lora_target_modules", "lora_num_layers", "batch_size", "max_steps",
    "lr_scheduler_type", "warmup_ratio",
)


def validate_boundary_selection(config, selection):
    selected = dict(selection.get("boundary_best") or {})
    require_values(selected, ("learning_rate", "dp_clip", "zo_eps"), "boundary_best")
    grid = config["stages"]["boundary"]["grid"]
    for key in ("learning_rate", "dp_clip", "zo_eps"):
        if selected[key] not in grid[key]:
            raise ValueError(f"boundary_best.{key} is not a member of the boundary grid")
    return selected


def validate_structure_selection(config, selection):
    boundary = validate_boundary_selection(config, selection)
    selected = dict(selection.get("structure_best") or {})
    keys = (
        "learning_rate", "dp_clip", "zo_eps", "lora_r", "lora_alpha",
        "lora_target_modules", "lora_num_layers",
    )
    require_values(selected, keys, "structure_best")
    if any(selected[key] != boundary[key] for key in boundary):
        raise ValueError("structure_best must retain the selected boundary values")
    stage = config["stages"]["structure"]
    variant = {
        "lora_target_modules": selected["lora_target_modules"],
        "lora_num_layers": selected["lora_num_layers"],
    }
    if variant not in stage["variants"]:
        raise ValueError("structure_best target modules/layers are not in the structure grid")
    fixed = stage["fixed"]
    if selected["lora_r"] != fixed["lora_r"] or selected["lora_alpha"] != fixed["lora_alpha"]:
        raise ValueError("structure_best must retain the fixed LoRA rank and alpha")
    return selected


def validate_finalists(config, selection):
    structure = validate_structure_selection(config, selection)
    finalists = selection.get("finalists") or []
    if len(finalists) != 2:
        raise ValueError("finalists must contain exactly two complete schedule configurations")
    schedule = config["stages"]["schedule"]
    normalized = []
    inherited_keys = tuple(structure)
    for index, candidate in enumerate(finalists):
        candidate = dict(candidate)
        require_values(candidate, TUNING_KEYS, f"finalists[{index}]")
        if any(candidate[key] != structure[key] for key in inherited_keys):
            raise ValueError(f"finalists[{index}] must retain structure_best")
        if candidate["lr_scheduler_type"] not in schedule["grid"]["lr_scheduler_type"]:
            raise ValueError(f"finalists[{index}] has an unknown scheduler")
        if candidate["warmup_ratio"] not in schedule["grid"]["warmup_ratio"]:
            raise ValueError(f"finalists[{index}] has an unknown warmup_ratio")
        for key in ("batch_size", "max_steps"):
            if candidate[key] != schedule["fixed"][key]:
                raise ValueError(f"finalists[{index}].{key} must retain the schedule fixed value")
        normalized.append(candidate)
    if normalized[0] == normalized[1]:
        raise ValueError("The two finalists must be different")
    return normalized


def parameter_sets(config, selection, stage_name):
    stage = config["stages"][stage_name]
    fixed = dict(stage.get("fixed", {}))
    if stage_name == "boundary":
        grid = stage["grid"]
        return [
            fixed | {"learning_rate": lr, "dp_clip": clip, "zo_eps": zo_eps}
            for lr, clip, zo_eps in itertools.product(
                grid["learning_rate"], grid["dp_clip"], grid["zo_eps"]
            )
        ]
    if stage_name == "structure":
        return [fixed | validate_boundary_selection(config, selection) | dict(variant)
                for variant in stage["variants"]]
    if stage_name == "schedule":
        inherited = validate_structure_selection(config, selection)
        grid = stage["grid"]
        return [
            fixed | inherited | {"lr_scheduler_type": scheduler, "warmup_ratio": warmup}
            for scheduler, warmup in itertools.product(
                grid["lr_scheduler_type"], grid["warmup_ratio"]
            )
        ]
    if stage_name == "confirm":
        return [dict(candidate) | {"seed": seed}
                for candidate, seed in itertools.product(
                    validate_finalists(config, selection), stage["seeds"]
                )]
    raise ValueError(f"Unknown stage {stage_name!r}")


def tag(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace(",", "+")


def expand_stage(config, selection, stage_name):
    jobs = []
    for ordinal, parameters in enumerate(parameter_sets(config, selection, stage_name)):
        common = dict(config["common"]) | parameters
        require_values(common, TUNING_KEYS + ("seed",), f"{stage_name} job")
        experiment_id = (
            f"lora-zo_dpga-explore-{stage_name}-{ordinal:02d}"
            f"-lr{tag(common['learning_rate'])}-clip{tag(common['dp_clip'])}"
            f"-mu{tag(common['zo_eps'])}-r{common['lora_r']}-a{common['lora_alpha']}"
            f"-targets{tag(common['lora_target_modules'])}-layers{common['lora_num_layers']}"
            f"-{common['lr_scheduler_type']}-wu{tag(common['warmup_ratio'])}"
            f"-bs{common['batch_size']}-steps{common['max_steps']}-seed{common['seed']}"
        )
        output_dir = Path(config["output_root"]) / stage_name / experiment_id
        jobs.append({
            "suite": f"lora_zo_dpga_explore_{stage_name}",
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
                "lora_target_modules": common["lora_target_modules"],
                "lora_num_layers": common["lora_num_layers"],
            },
        })
    return jobs


def execute_job(job, config_path, selection_path, resume=False):
    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output exists: {output_dir}. Use --resume only for an intended resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(job)
    snapshot = dict(job)
    snapshot.update({
        "output_dir": str(job["output_dir"]),
        "command": command,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    })
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != command:
            raise ValueError("Existing run_config.json does not match the requested resume command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=LLAMA_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run round-two LoRA + ZODPGA exploration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run", action="store_true")
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
        raise ValueError("Execution requires --index N")
    execute_job(selected[0], args.config, args.selection, resume=args.resume)


if __name__ == "__main__":
    main()
