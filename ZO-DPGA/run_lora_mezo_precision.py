import argparse
import itertools
import shlex
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml, require_values
from run_lora_mezo_exploration import execute_job, tag


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_mezo_precision.yaml"
DEFAULT_SELECTION = LLAMA_DIR / "configs" / "lora_mezo_precision_selection.yaml"
STAGES = ("optimizer", "layers", "scale", "batch", "confirm")
CONFIG_KEYS = (
    "learning_rate", "zo_eps", "lora_r", "lora_alpha",
    "lora_target_modules", "lora_num_layers", "batch_size", "max_steps",
    "lr_scheduler_type", "warmup_ratio",
)


def validate_optimizer(config, selection):
    value = dict(selection.get("optimizer_best") or {})
    require_values(value, ("learning_rate", "zo_eps"), "optimizer_best")
    grid = config["stages"]["optimizer"]["grid"]
    for key in ("learning_rate", "zo_eps"):
        if value[key] not in grid[key]:
            raise ValueError(f"optimizer_best.{key} is outside the optimizer grid")
    return value


def validate_layers(config, selection):
    inherited = validate_optimizer(config, selection)
    value = dict(selection.get("layers_best") or {})
    keys = ("learning_rate", "zo_eps", "lora_target_modules", "lora_num_layers")
    require_values(value, keys, "layers_best")
    if any(value[key] != inherited[key] for key in inherited):
        raise ValueError("layers_best must retain optimizer_best")
    stage = config["stages"]["layers"]
    if value["lora_target_modules"] != stage["fixed"]["lora_target_modules"]:
        raise ValueError("layers_best must retain q_proj,k_proj,v_proj")
    if value["lora_num_layers"] not in stage["layer_values"]:
        raise ValueError("layers_best.lora_num_layers is outside the layers grid")
    return value


def validate_scale(config, selection):
    inherited = validate_layers(config, selection)
    value = dict(selection.get("scale_best") or {})
    keys = tuple(inherited) + ("lora_r", "lora_alpha")
    require_values(value, keys, "scale_best")
    if any(value[key] != inherited[key] for key in inherited):
        raise ValueError("scale_best must retain layers_best")
    variant = {key: value[key] for key in ("lora_r", "lora_alpha")}
    if variant not in config["stages"]["scale"]["variants"]:
        raise ValueError("scale_best is outside the scale variants")
    return value


def validate_finalists(config, selection):
    inherited = validate_scale(config, selection)
    values = [dict(item) for item in (selection.get("finalists") or [])]
    if len(values) != 2:
        raise ValueError("finalists must contain exactly two configurations")
    stage = config["stages"]["batch"]
    for index, value in enumerate(values):
        require_values(value, CONFIG_KEYS, f"finalists[{index}]")
        if any(value[key] != inherited[key] for key in inherited):
            raise ValueError(f"finalists[{index}] must retain scale_best")
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
    if stage_name == "optimizer":
        grid = stage["grid"]
        return [
            fixed | {"learning_rate": learning_rate, "zo_eps": zo_eps}
            for learning_rate, zo_eps in itertools.product(
                grid["learning_rate"], grid["zo_eps"]
            )
        ]
    if stage_name == "layers":
        inherited = validate_optimizer(config, selection)
        return [fixed | inherited | {"lora_num_layers": value}
                for value in stage["layer_values"]]
    if stage_name == "scale":
        inherited = validate_layers(config, selection)
        return [fixed | inherited | dict(value) for value in stage["variants"]]
    if stage_name == "batch":
        inherited = validate_scale(config, selection)
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
            f"lora-mezo-precision-{stage_name}-{ordinal:02d}"
            f"-lr{tag(common['learning_rate'])}-mu{tag(common['zo_eps'])}"
            f"-r{common['lora_r']}-a{common['lora_alpha']}"
            f"-targets{tag(common['lora_target_modules'])}-layers{common['lora_num_layers']}"
            f"-bs{common['batch_size']}-steps{common['max_steps']}"
            f"-{common['lr_scheduler_type']}-seed{common['seed']}"
        )
        output_dir = Path(config["output_root"]) / stage_name / experiment_id
        jobs.append({
            "suite": f"lora_mezo_precision_{stage_name}",
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
            "force_lora_structure_args": True,
            "mode_config": {key: common[key] for key in (
                "lora_r", "lora_alpha", "lora_target_modules", "lora_num_layers"
            )},
        })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Third-round LoRA + MeZO precision search")
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
