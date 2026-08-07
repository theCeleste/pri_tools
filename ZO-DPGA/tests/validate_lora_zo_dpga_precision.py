import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for  # noqa: E402
from run_lora_zo_dpga_matrix import load_yaml  # noqa: E402
from run_lora_zo_dpga_precision import expand_stage  # noqa: E402


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_zo_dpga_precision.yaml")
    selection = {
        "version": 1,
        "core_best": {"zo_eps": 9.0e-3, "lora_num_layers": 16, "max_steps": 10000},
        "privacy_best": {
            "learning_rate": 1.0e-4, "dp_clip": 10.0, "zo_eps": 9.0e-3,
            "lora_num_layers": 16, "max_steps": 10000,
        },
        "rank_best": {
            "learning_rate": 1.0e-4, "dp_clip": 10.0, "zo_eps": 9.0e-3,
            "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
            "lora_r": 12, "lora_alpha": 24, "max_steps": 10000,
        },
        "finalists": [
            {
                "learning_rate": 1.0e-4, "dp_clip": 10.0, "zo_eps": 9.0e-3,
                "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
                "lora_r": 12, "lora_alpha": 24, "batch_size": 4,
                "max_steps": 10000, "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            },
            {
                "learning_rate": 1.0e-4, "dp_clip": 10.0, "zo_eps": 9.0e-3,
                "lora_target_modules": "q_proj,v_proj", "lora_num_layers": 16,
                "lora_r": 12, "lora_alpha": 24, "batch_size": 6,
                "max_steps": 10000, "lr_scheduler_type": "constant", "warmup_ratio": 0.0,
            },
        ],
    }
    expected = {"core": 12, "privacy_update": 9, "rank": 3, "batch": 4, "confirm": 6}
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
                "--trainer", "--zo_dpga", "--dp_epsilon", "--dp_delta",
                "--zo_dpga_clip_threshold", "--dev_only", "--lora",
                "--lora_target_modules", "--lora_num_layers",
            ):
                if flag not in command:
                    raise AssertionError(f"Missing {flag}: {command}")
            if command[command.index("--trainer") + 1] != "zo":
                raise AssertionError(f"Expected zeroth-order trainer: {command}")
            if command[command.index("--dp_epsilon") + 1] != "6.0":
                raise AssertionError("ZODPGA epsilon changed")
            if command[command.index("--dp_delta") + 1] != "1e-05":
                raise AssertionError("ZODPGA delta changed")
            if job["common"]["gradient_accumulation_steps"] != 1 or not job["dev_only"]:
                raise AssertionError("ZODPGA training/dev invariants changed")
    print(f"PASS LoRA ZODPGA precision matrix: {len(identities)} unique jobs")


if __name__ == "__main__":
    main()
