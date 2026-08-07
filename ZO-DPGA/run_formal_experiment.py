import argparse
import hashlib
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


LLAMA_DIR = Path(__file__).resolve().parent


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError(f"Unsupported experiment config version: {config.get('version')}")
    return config


def expand_suite(config, suite_name):
    suite = config["suites"][suite_name]
    common = dict(config["common"])
    common.update(suite.get("overrides", {}))
    epsilons = suite.get("dp_epsilons", [None])
    jobs = []
    mode_overrides = suite.get("mode_overrides", {})
    unknown_modes = set(mode_overrides) - set(suite["modes"])
    if unknown_modes:
        raise ValueError(
            f"Suite {suite_name} has overrides for modes it does not run: {sorted(unknown_modes)}"
        )
    grid = []
    for mode in suite["modes"]:
        mode_common = dict(common)
        mode_common.update(mode_overrides.get(mode, {}))
        learning_rates = suite.get("learning_rates", [mode_common["learning_rate"]])
        zo_eps_values = suite.get("zo_eps_values", [mode_common["zo_eps"]])
        clip_values = suite.get("dp_clips", [mode_common["dp_clip"]])
        batch_sizes = suite.get("batch_sizes", [mode_common["batch_size"]])
        for values in itertools.product(
            suite["seeds"], epsilons,
            learning_rates, zo_eps_values, clip_values, batch_sizes,
        ):
            grid.append((mode, mode_common, *values))
    for mode, mode_common, seed, epsilon, learning_rate, zo_eps, clip, batch_size in grid:
        method = suite["method"]
        if method != "zo_dpga" and epsilon is not None:
            raise ValueError(f"Suite {suite_name} sets epsilon for non-DP method {method}")
        epsilon_tag = f"-eps{epsilon:g}" if epsilon is not None else ""
        tuning_tags = ""
        if "learning_rates" in suite:
            tuning_tags += f"-lr{learning_rate:g}"
        if "zo_eps_values" in suite:
            tuning_tags += f"-mu{zo_eps:g}"
        if "dp_clips" in suite:
            tuning_tags += f"-clip{clip:g}"
        if "batch_sizes" in suite:
            tuning_tags += f"-bs{batch_size}"
        experiment_id = f"{suite_name}-{method}-{mode}{epsilon_tag}{tuning_tags}-seed{seed}"
        output_dir = Path(config["output_root"]) / suite_name / experiment_id
        job_common = dict(mode_common)
        job_common.update({
            "learning_rate": learning_rate,
            "zo_eps": zo_eps,
            "dp_clip": clip,
            "batch_size": batch_size,
        })
        jobs.append({
            "suite": suite_name,
            "experiment_id": experiment_id,
            "description": suite["description"],
            "privacy_accounting_note": config.get("privacy_accounting_note"),
            "model_name": config["model_name"],
            "method": method,
            "mode": mode,
            "seed": seed,
            "dp_epsilon": epsilon,
            "dev_only": bool(suite.get("dev_only", False)),
            "output_dir": output_dir,
            "common": job_common,
            "mode_config": config["modes"][mode],
        })
    return jobs


def command_for(job):
    common = job["common"]
    command = [
        sys.executable,
        "run.py",
        "--model_name", job["model_name"],
        "--task_name", common["task_name"],
        "--output_dir", str(job["output_dir"]),
        "--result_file", str(job["output_dir"] / "eval_metrics.json"),
        "--experiment_suite", job["suite"],
        "--experiment_id", job["experiment_id"],
        "--tag", job["experiment_id"],
        "--seed", str(job["seed"]),
        "--train_set_seed", str(job["seed"]),
    ]
    if common.get("load_bfloat16"):
        command.append("--load_bfloat16")

    if job["method"] == "zero_shot":
        command.extend([
            "--trainer", "none",
            "--num_train", str(common["num_train"]),
            "--num_eval", str(common["num_eval"]),
        ])
        if job.get("dev_only"):
            command.extend(["--num_dev", str(common["num_dev"]), "--dev_only"])
        return command

    command.extend([
        "--num_train", str(common["num_train"]),
        "--num_dev", str(common["num_dev"]),
        "--num_eval", str(common["num_eval"]),
        "--max_length", str(common["max_length"]),
        "--max_steps", str(common["max_steps"]),
        "--per_device_train_batch_size", str(common["batch_size"]),
        "--gradient_accumulation_steps", str(common["gradient_accumulation_steps"]),
        "--learning_rate", str(common["learning_rate"]),
        "--zo_eps", str(common["zo_eps"]),
        "--lr_scheduler_type", str(common.get("lr_scheduler_type", "constant")),
        "--logging_steps", str(common["logging_steps"]),
        "--evaluation_strategy", "steps",
        "--eval_steps", str(common["eval_steps"]),
        "--save_strategy", "steps",
        "--save_steps", str(common["eval_steps"]),
        "--save_total_limit", "1",
        "--load_best_model_at_end",
    ])
    if "warmup_ratio" in common:
        command.extend(["--warmup_ratio", str(common["warmup_ratio"])])
    if "weight_decay" in common:
        command.extend(["--weight_decay", str(common["weight_decay"])])
    if common.get("train_as_classification"):
        command.append("--train_as_classification")
    if job.get("dev_only"):
        command.append("--dev_only")

    if job["method"] == "zo_dpga":
        command.extend([
            "--trainer", "zo",
            "--zo_dpga",
            "--dp_epsilon", str(job["dp_epsilon"]),
            "--dp_delta", str(common["dp_delta"]),
            "--zo_dpga_clip_threshold", str(common["dp_clip"]),
        ])
    elif job["method"] == "mezo":
        command.extend(["--trainer", "zo"])
    else:
        raise ValueError(f"Unsupported formal method: {job['method']}")

    mode_config = job["mode_config"]
    if job["mode"] == "lora":
        command.extend([
            "--lora",
            "--lora_r", str(mode_config["lora_r"]),
            "--lora_alpha", str(mode_config["lora_alpha"]),
        ])
        target_modules = mode_config.get("lora_target_modules", "q_proj,v_proj")
        num_layers = mode_config.get("lora_num_layers", -1)
        force_structure_args = bool(job.get("force_lora_structure_args", False))
        if force_structure_args or target_modules != "q_proj,v_proj":
            command.extend(["--lora_target_modules", str(target_modules)])
        if force_structure_args or num_layers != -1:
            command.extend(["--lora_num_layers", str(num_layers)])
    elif job["mode"] == "prefix":
        command.extend(["--prefix_tuning", "--num_prefix", str(mode_config["num_prefix"]), "--no_reparam"])
        if mode_config.get("prefix_init_by_real_act"):
            command.append("--prefix_init_by_real_act")
    elif job["mode"] == "head":
        command.append("--head_tuning")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=LLAMA_DIR / "configs" / "formal_experiments.yaml")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run", action="store_true", help="Execute one selected job; default is dry-run")
    parser.add_argument("--resume", action="store_true", help="Allow an existing output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.suite not in config["suites"]:
        raise ValueError(f"Unknown suite {args.suite!r}; choose from {sorted(config['suites'])}")
    jobs = expand_suite(config, args.suite)
    selected = jobs if args.index is None else [jobs[args.index]]

    print(f"Suite {args.suite}: {len(jobs)} jobs")
    for index, job in enumerate(jobs):
        marker = "*" if job in selected else " "
        print(f"{marker} [{index:02d}] {job['experiment_id']} -> {job['output_dir']}")

    if not args.run:
        if args.index is not None:
            print("Command:", shlex.join(command_for(selected[0])))
        print("Dry run only. Add --index N --run to execute exactly one job.")
        return
    if args.index is None:
        raise ValueError("Execution requires --index N; bulk execution is intentionally disabled")

    job = selected[0]
    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {output_dir}. Use --resume only for an intended resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(job)
    snapshot["output_dir"] = str(job["output_dir"])
    snapshot["command"] = command_for(job)
    snapshot["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != snapshot["command"]:
            raise ValueError("Existing run_config.json does not match the requested resume command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command_for(job), cwd=LLAMA_DIR, check=True)


if __name__ == "__main__":
    main()
