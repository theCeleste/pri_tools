import argparse
import itertools
import shlex
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml, require_values
from run_lora_mezo_exploration import execute_job, tag


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_zo_dpga_precision.yaml"
DEFAULT_SELECTION = LLAMA_DIR / "configs" / "lora_zo_dpga_precision_selection.yaml"
STAGES = ("core", "privacy_update", "rank", "batch", "confirm")
CONFIG_KEYS = (
    "learning_rate", "dp_clip", "zo_eps", "lora_r", "lora_alpha",
    "lora_target_modules", "lora_num_layers", "batch_size", "max_steps",
    "lr_scheduler_type", "warmup_ratio",
)


def validate_core(config, selection):
    value = dict(selection.get("core_best") or {})
    keys = ("zo_eps", "lora_num_layers", "max_steps")
    require_values(value, keys, "core_best")
    grid = config["stages"]["core"]["grid"]
    if any(value[key] not in grid[key] for key in keys):
        raise ValueError("core_best is outside the core grid")
    return value


def validate_privacy(config, selection):
    inherited = validate_core(config, selection)
    value = dict(selection.get("privacy_best") or {})
    keys = ("learning_rate", "dp_clip") + tuple(inherited)
    require_values(value, keys, "privacy_best")
    if any(value[key] != inherited[key] for key in inherited):
        raise ValueError("privacy_best must retain core_best")
    grid = config["stages"]["privacy_update"]["grid"]
    if value["learning_rate"] not in grid["learning_rate"] or value["dp_clip"] not in grid["dp_clip"]:
        raise ValueError("privacy_best LR/clip is outside the privacy_update grid")
    return value


def validate_rank(config, selection):
    inherited = validate_privacy(config, selection)
    value = dict(selection.get("rank_best") or {})
    keys = tuple(inherited) + (
        "lora_target_modules", "lora_r", "lora_alpha"
    )
    require_values(value, keys, "rank_best")
    if any(value[key] != inherited[key] for key in inherited):
        raise ValueError("rank_best must retain privacy_best")
    if value["lora_target_modules"] != config["stages"]["rank"]["fixed"]["lora_target_modules"]:
        raise ValueError("rank_best must retain q_proj,v_proj")
    variant = {key: value[key] for key in ("lora_r", "lora_alpha")}
    if variant not in config["stages"]["rank"]["variants"]:
        raise ValueError("rank_best is outside the rank variants")
    return value


def validate_finalists(config, selection):
    inherited = validate_rank(config, selection)
    values = [dict(item) for item in (selection.get("finalists") or [])]
    if len(values) != 2:
        raise ValueError("finalists must contain exactly two ZODPGA configurations")
    stage = config["stages"]["batch"]
    for index, value in enumerate(values):
        require_values(value, CONFIG_KEYS, f"finalists[{index}]")
        if any(value[key] != inherited[key] for key in inherited):
            raise ValueError(f"finalists[{index}] must retain rank_best")
        if value["batch_size"] not in stage["batch_sizes"]:
            raise ValueError(f"finalists[{index}].batch_size is outside the batch grid")
        for key, fixed in stage["fixed"].items():
            if key != "seed" and value[key] != fixed:
                raise ValueError(f"finalists[{index}].{key} must retain the batch fixed value")
    if values[0] == values[1]:
        raise ValueError("The finalists must differ")
    return values


def parameter_sets(config, selection, stage_name):
    stage = config["stages"][stage_name]
    fixed = dict(stage.get("fixed", {}))
    if stage_name == "core":
        grid = stage["grid"]
        return [
            fixed | {"zo_eps": zo_eps, "lora_num_layers": layers, "max_steps": steps}
            for zo_eps, layers, steps in itertools.product(
                grid["zo_eps"], grid["lora_num_layers"], grid["max_steps"]
            )
        ]
    if stage_name == "privacy_update":
        inherited = validate_core(config, selection)
        grid = stage["grid"]
        return [
            fixed | inherited | {"learning_rate": lr, "dp_clip": clip}
            for lr, clip in itertools.product(grid["learning_rate"], grid["dp_clip"])
        ]
    if stage_name == "rank":
        inherited = validate_privacy(config, selection)
        return [fixed | inherited | dict(value) for value in stage["variants"]]
    if stage_name == "batch":
        inherited = validate_rank(config, selection)
        return [fixed | inherited | {"batch_size": value} for value in stage["batch_sizes"]]
    if stage_name == "confirm":
        return [dict(item) | {"seed": seed} for item, seed in itertools.product(
            validate_finalists(config, selection), stage["seeds"]
        )]
    raise ValueError(f"Unknown stage {stage_name!r}")


def expand_stage(config, selection, stage_name):
    jobs = []
    for ordinal, parameters in enumerate(parameter_sets(config, selection, stage_name)):
        common = dict(config["common"]) | parameters
        require_values(common, CONFIG_KEYS + ("seed",), f"{stage_name} job")
        experiment_id = (
            f"lora-zo_dpga-precision-{stage_name}-{ordinal:02d}"
            f"-lr{tag(common['learning_rate'])}-clip{tag(common['dp_clip'])}"
            f"-mu{tag(common['zo_eps'])}-r{common['lora_r']}-a{common['lora_alpha']}"
            f"-targets{tag(common['lora_target_modules'])}-layers{common['lora_num_layers']}"
            f"-bs{common['batch_size']}-steps{common['max_steps']}"
            f"-{common['lr_scheduler_type']}-seed{common['seed']}"
        )
        output_dir = Path(config["output_root"]) / stage_name / experiment_id
        jobs.append({
            "suite": f"lora_zo_dpga_precision_{stage_name}",
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
            "force_lora_structure_args": True,
            "mode_config": {key: common[key] for key in (
                "lora_r", "lora_alpha", "lora_target_modules", "lora_num_layers"
            )},
        })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="ZODPGA-only precision search at fixed privacy budget")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, selection = load_yaml(args.config), load_yaml(args.selection)
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
