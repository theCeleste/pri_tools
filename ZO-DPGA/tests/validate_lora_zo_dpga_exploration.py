import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for  # noqa: E402
from run_lora_zo_dpga_exploration import expand_stage, load_yaml  # noqa: E402


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_zo_dpga_exploration.yaml")
    selection = {
        "version": 1,
        "boundary_best": {"learning_rate": 2.0e-4, "dp_clip": 20.0, "zo_eps": 6.0e-3},
        "structure_best": {
            "learning_rate": 2.0e-4, "dp_clip": 20.0, "zo_eps": 6.0e-3,
            "lora_r": 8, "lora_alpha": 16,
            "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
        },
        "finalists": [
            {
                "learning_rate": 2.0e-4, "dp_clip": 20.0, "zo_eps": 6.0e-3,
                "lora_r": 8, "lora_alpha": 16,
                "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
                "batch_size": 4, "max_steps": 10000,
                "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            },
            {
                "learning_rate": 2.0e-4, "dp_clip": 20.0, "zo_eps": 6.0e-3,
                "lora_r": 8, "lora_alpha": 16,
                "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
                "batch_size": 4, "max_steps": 10000,
                "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
            },
        ],
    }
    expected = {"boundary": 12, "structure": 6, "schedule": 6, "confirm": 6}
    identities = set()
    for stage, count in expected.items():
        jobs = expand_stage(config, selection, stage)
        if len(jobs) != count:
            raise AssertionError(f"{stage}: expected {count}, got {len(jobs)}")
        for job in jobs:
            if job["experiment_id"] in identities:
                raise AssertionError(f"Duplicate experiment: {job['experiment_id']}")
            identities.add(job["experiment_id"])
            command = command_for(job)
            required = [
                "--zo_dpga", "--dev_only", "--lora", "--warmup_ratio",
                "--lr_scheduler_type",
            ]
            if any(flag not in command for flag in required):
                raise AssertionError(f"Incomplete command: {command}")
            mode = job["mode_config"]
            effective_targets = (
                command[command.index("--lora_target_modules") + 1]
                if "--lora_target_modules" in command else "q_proj,v_proj"
            )
            effective_layers = (
                int(command[command.index("--lora_num_layers") + 1])
                if "--lora_num_layers" in command else -1
            )
            if effective_targets != mode["lora_target_modules"] or effective_layers != mode["lora_num_layers"]:
                raise AssertionError(f"LoRA structure flags do not match the job: {command}")
            if job["dp_epsilon"] != 6.0 or job["common"]["gradient_accumulation_steps"] != 1:
                raise AssertionError("Privacy/training invariants changed")
    print(f"PASS exploration matrix: {len(identities)} unique jobs")


if __name__ == "__main__":
    main()
