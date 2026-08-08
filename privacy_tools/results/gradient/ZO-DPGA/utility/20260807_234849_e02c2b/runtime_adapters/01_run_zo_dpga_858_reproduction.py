import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "zo_dpga_858_reproduction.yaml"


def build_job(config, output_dir=None):
    run = dict(config["run"])
    if run["train_set_seed"] != run["seed"]:
        raise ValueError("The frozen reproduction requires train_set_seed == seed == 2")
    output_dir = Path(output_dir or config["output_dir"])
    common = {
        "task_name": run["task_name"],
        "num_train": run["num_train"],
        "num_dev": run["num_dev"],
        "num_eval": run["num_eval"],
        "max_length": run["max_length"],
        "max_steps": run["max_steps"],
        "batch_size": run["batch_size"],
        "gradient_accumulation_steps": run["gradient_accumulation_steps"],
        "learning_rate": run["learning_rate"],
        "zo_eps": run["zo_eps"],
        "lr_scheduler_type": run["lr_scheduler_type"],
        "warmup_ratio": run["warmup_ratio"],
        "weight_decay": run["weight_decay"],
        "logging_steps": run["logging_steps"],
        "eval_steps": run["eval_steps"],
        "load_bfloat16": run["load_bfloat16"],
        "train_as_classification": run["train_as_classification"],
        "dp_delta": run["dp_delta"],
        "dp_clip": run["dp_clip"],
    }
    return {
        "suite": config["experiment_suite"],
        "experiment_id": config["experiment_id"],
        "description": "Frozen reproduction of the 85.8% seed-2 dev run",
        "model_name": "meta-llama/Llama-2-7b-hf",
        "method": "zo_dpga",
        "mode": "lora",
        "seed": run["seed"],
        "dp_epsilon": run["dp_epsilon"],
        "dev_only": run["dev_only"],
        "force_lora_structure_args": True,
        "output_dir": output_dir,
        "common": common,
        "mode_config": {
            "lora_r": run["lora_r"],
            "lora_alpha": run["lora_alpha"],
            "lora_target_modules": run["lora_target_modules"],
            "lora_num_layers": run["lora_num_layers"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce the frozen 85.8% ZO-DPGA run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory for an independent repeat")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    job = build_job(config, args.output_dir)
    command = command_for(job)
    print("Reference:", json.dumps(config["expected_reference"], ensure_ascii=False))
    print("Output:", job["output_dir"])
    print("Command:", shlex.join(command))
    if not args.run:
        print("Dry run only. Add --run to execute.")
        return

    output_dir = LLAMA_DIR / job["output_dir"]
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output exists: {output_dir}. Choose a fresh --output-dir for an independent reproduction."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(job)
    snapshot.update({
        "output_dir": str(job["output_dir"]),
        "command": command,
        "expected_reference": config["expected_reference"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    })
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != command:
            raise ValueError("Existing run_config.json does not match the frozen reproduction command")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=LLAMA_DIR, check=True)


if __name__ == "__main__":
    main()
