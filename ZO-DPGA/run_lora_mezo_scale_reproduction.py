import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import LLAMA_DIR, load_yaml


DEFAULT_CONFIG = LLAMA_DIR / "configs" / "lora_mezo_scale_reproductions.yaml"


def build_job(config, variant_name, output_dir=None):
    if variant_name not in config["variants"]:
        raise ValueError(f"Unknown variant {variant_name!r}; choose from {sorted(config['variants'])}")
    run = dict(config["common"])
    variant = config["variants"][variant_name]
    if run["train_set_seed"] != run["seed"]:
        raise ValueError("Frozen MeZO reproductions require train_set_seed == seed")
    output_dir = Path(output_dir or Path(config["output_root"]) / variant_name)
    return {
        "suite": config["experiment_suite"],
        "experiment_id": variant["experiment_id"],
        "description": f"Frozen reproduction of {variant_name}",
        "model_name": config["model_name"],
        "method": "mezo",
        "mode": "lora",
        "seed": run["seed"],
        "dp_epsilon": None,
        "dev_only": run["dev_only"],
        "force_lora_structure_args": True,
        "output_dir": output_dir,
        "common": {key: run[key] for key in (
            "task_name", "num_train", "num_dev", "num_eval", "max_length",
            "max_steps", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "zo_eps", "lr_scheduler_type", "warmup_ratio",
            "weight_decay", "logging_steps", "eval_steps", "load_bfloat16",
            "train_as_classification",
        )},
        "mode_config": {
            "lora_r": variant["lora_r"],
            "lora_alpha": variant["lora_alpha"],
            "lora_target_modules": run["lora_target_modules"],
            "lora_num_layers": run["lora_num_layers"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce frozen LoRA + MeZO scale runs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory for an independent repeat")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    job = build_job(config, args.variant, args.output_dir)
    reference = config["variants"][args.variant]["expected_reference"]
    command = command_for(job)
    print("Variant:", args.variant)
    print("Reference:", json.dumps(reference, ensure_ascii=False))
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
        "variant": args.variant,
        "output_dir": str(job["output_dir"]),
        "command": command,
        "expected_reference": reference,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    })
    snapshot_file = output_dir / "run_config.json"
    if snapshot_file.exists():
        existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        if existing.get("command") != command or existing.get("variant") != args.variant:
            raise ValueError("Existing run_config.json does not match the frozen reproduction")
    else:
        snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=LLAMA_DIR, check=True)


if __name__ == "__main__":
    main()
