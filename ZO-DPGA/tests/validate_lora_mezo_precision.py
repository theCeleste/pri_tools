import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for  # noqa: E402
from run_lora_zo_dpga_matrix import load_yaml  # noqa: E402
from run_lora_mezo_precision import expand_stage  # noqa: E402


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_precision.yaml")
    selection = {
        "version": 1,
        "optimizer_best": {"learning_rate": 2.0e-4, "zo_eps": 6.0e-3},
        "layers_best": {
            "learning_rate": 2.0e-4, "zo_eps": 6.0e-3,
            "lora_target_modules": "q_proj,k_proj,v_proj", "lora_num_layers": 20,
        },
        "scale_best": {
            "learning_rate": 2.0e-4, "zo_eps": 6.0e-3,
            "lora_target_modules": "q_proj,k_proj,v_proj", "lora_num_layers": 20,
            "lora_r": 16, "lora_alpha": 32,
        },
        "finalists": [
            {
                "learning_rate": 2.0e-4, "zo_eps": 6.0e-3,
                "lora_target_modules": "q_proj,k_proj,v_proj", "lora_num_layers": 20,
                "lora_r": 16, "lora_alpha": 32, "batch_size": 8,
                "max_steps": 15000, "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            },
            {
                "learning_rate": 2.0e-4, "zo_eps": 6.0e-3,
                "lora_target_modules": "q_proj,k_proj,v_proj", "lora_num_layers": 20,
                "lora_r": 16, "lora_alpha": 32, "batch_size": 12,
                "max_steps": 15000, "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            },
        ],
    }
    expected = {"optimizer": 12, "layers": 5, "scale": 6, "batch": 3, "confirm": 6}
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
            for flag in (
                "--trainer", "--dev_only", "--lora", "--lora_target_modules",
                "--lora_num_layers", "--train_set_seed",
            ):
                if flag not in command:
                    raise AssertionError(f"Missing {flag}: {command}")
            if command[command.index("--trainer") + 1] != "zo" or "--zo_dpga" in command:
                raise AssertionError(f"Expected non-private MeZO command: {command}")
            if job["common"]["gradient_accumulation_steps"] != 1 or not job["dev_only"]:
                raise AssertionError("Training/dev invariants changed")
    print(f"PASS LoRA MeZO precision matrix: {len(identities)} unique jobs")


if __name__ == "__main__":
    main()
