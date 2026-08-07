import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml, require_values


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_mezo_exploration.yaml"
DEFAULT_SELECTION = LLAMA_DIR / "configs" / "lora_mezo_exploration_selection.yaml"
STAGES = ("optimizer", "structure", "budget", "confirm")
CONFIG_KEYS = (
    "learning_rate", "zo_eps", "lora_r", "lora_alpha",
    "lora_target_modules", "lora_num_layers", "batch_size", "max_steps",
    "lr_scheduler_type", "warmup_ratio",
)


def validate_optimizer_selection(config, selection):
    selected = dict(selection.get("optimizer_best") or {})
    require_values(selected, ("learning_rate", "zo_eps"), "optimizer_best")
    grid = config["stages"]["optimizer"]["grid"]
    for key in ("learning_rate", "zo_eps"):
        if selected[key] not in grid[key]:
            raise ValueError(f"optimizer_best.{key} is not in the optimizer grid")
    return selected


def validate_structure_selection(config, selection):
    optimizer = validate_optimizer_selection(config, selection)
    selected = dict(selection.get("structure_best") or {})
    keys = (
        "learning_rate", "zo_eps", "lora_r", "lora_alpha",
        "lora_target_modules", "lora_num_layers",
    )
    require_values(selected, keys, "structure_best")
    if any(selected[key] != optimizer[key] for key in optimizer):
        raise ValueError("structure_best must retain optimizer_best")
    variant = {key: selected[key] for key in keys if key not in optimizer}
    if variant not in config["stages"]["structure"]["variants"]:
        raise ValueError("structure_best adapter is not in the structure variants")
    return selected


def validate_finalists(config, selection):
    structure = validate_structure_selection(config, selection)
    finalists = selection.get("finalists") or []
    if len(finalists) != 2:
        raise ValueError("finalists must contain exactly two configurations")
    stage = config["stages"]["budget"]
    normalized = []
    for index, candidate in enumerate(finalists):
        candidate = dict(candidate)
        require_values(candidate, CONFIG_KEYS, f"finalists[{index}]")
        if any(candidate[key] != structure[key] for key in structure):
            raise ValueError(f"finalists[{index}] must retain structure_best")
        budget = {
            key: candidate[key]
            for key in ("batch_size", "max_steps", "lr_scheduler_type")
        }
        if budget not in stage["variants"]:
            raise ValueError(f"finalists[{index}] is not in the budget variants")
        if candidate["warmup_ratio"] != stage["fixed"]["warmup_ratio"]:
            raise ValueError(f"finalists[{index}] has an unknown warmup ratio")
        normalized.append(candidate)
    if normalized[0] == normalized[1]:
        raise ValueError("The two finalists must differ")
    return normalized


def parameter_sets(config, selection, stage_name):
    stage = config["stages"][stage_name]
    fixed = dict(stage.get("fixed", {}))
    if stage_name == "optimizer":
        grid = stage["grid"]
        return [
            fixed | {"learning_rate": lr, "zo_eps": eps}
            for lr, eps in itertools.product(grid["learning_rate"], grid["zo_eps"])
        ]
    if stage_name == "structure":
        inherited = validate_optimizer_selection(config, selection)
        return [fixed | inherited | dict(variant) for variant in stage["variants"]]
    if stage_name == "budget":
        inherited = validate_structure_selection(config, selection)
        return [fixed | inherited | dict(variant) for variant in stage["variants"]]
    if stage_name == "confirm":
        return [
            dict(candidate) | {"seed": seed}
            for candidate, seed in itertools.product(
                validate_finalists(config, selection), stage["seeds"]
            )
        ]
    raise ValueError(f"Unknown stage {stage_name!r}")


def tag(value):
    return f"{value:g}" if isinstance(value, float) else str(value).replace(",", "+")


def expand_stage(config, selection, stage_name):
    jobs = []
    for ordinal, parameters in enumerate(parameter_sets(config, selection, stage_name)):
        common = dict(config["common"]) | parameters
        require_values(common, CONFIG_KEYS + ("seed",), f"{stage_name} job")
        experiment_id = (
            f"lora-mezo-{stage_name}-{ordinal:02d}"
            f"-lr{tag(common['learning_rate'])}-mu{tag(common['zo_eps'])}"
            f"-r{common['lora_r']}-a{common['lora_alpha']}"
            f"-targets{tag(common['lora_target_modules'])}-layers{common['lora_num_layers']}"
            f"-bs{common['batch_size']}-steps{common['max_steps']}"
            f"-{common['lr_scheduler_type']}-seed{common['seed']}"
        )
        output_dir = Path(config["output_root"]) / stage_name / experiment_id
        jobs.append({
            "suite": f"lora_mezo_{stage_name}",
            "experiment_id": experiment_id,
            "description": config["stages"][stage_name]["description"],
            "model_name": config["model_name"],
            "method": "mezo",
            "mode": "lora",
            "seed": common["seed"],
            "dp_epsilon": None,
            "dev_only": bool(common["dev_only"]),
            "output_dir": output_dir,
            "common": common,
            "mode_config": {
                "lora_r": common["lora_r"],
                "lora_alpha": common["lora_alpha"],
                "lora_target_modules": common["lora_target_modules"],
                "lora_num_layers": common["lora_num_layers"],
            },
            "force_lora_structure_args": True,
        })
    return jobs


def execute_job(job, config_path, selection_path, resume=False):
    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output exists: {output_dir}. Use --resume only intentionally.")
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
            raise ValueError("Existing run_config.json does not match the resume command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=LLAMA_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run development-only LoRA + MeZO exploration")
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
