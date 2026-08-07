import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for  # noqa: E402
from run_lora_zo_dpga_matrix import expand_stage, load_yaml  # noqa: E402


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_zo_dpga_matrix.yaml")
    selection = {
        "version": 1,
        "stage1_best": {"learning_rate": 1.0e-5, "dp_clip": 0.1},
        "stage2_best": {
            "learning_rate": 1.0e-5,
            "dp_clip": 0.1,
            "lora_r": 4,
            "lora_alpha": 8,
            "zo_eps": 1.0e-3,
        },
        "finalists": [
            {
                "learning_rate": 1.0e-5, "dp_clip": 0.1,
                "lora_r": 4, "lora_alpha": 8, "zo_eps": 1.0e-3,
                "batch_size": 4, "max_steps": 10000,
            },
            {
                "learning_rate": 1.0e-5, "dp_clip": 0.1,
                "lora_r": 4, "lora_alpha": 8, "zo_eps": 1.0e-3,
                "batch_size": 8, "max_steps": 5000,
            },
        ],
    }
    expected = {"stage1": 16, "stage2": 12, "stage3": 3, "final": 6}
    all_ids = set()
    for stage, count in expected.items():
        jobs = expand_stage(config, selection, stage)
        if len(jobs) != count:
            raise AssertionError(f"{stage}: expected {count} jobs, got {len(jobs)}")
        for job in jobs:
            if job["experiment_id"] in all_ids:
                raise AssertionError(f"Duplicate experiment ID: {job['experiment_id']}")
            all_ids.add(job["experiment_id"])
            command = command_for(job)
            required = [
                "--trainer", "--zo_dpga", "--dev_only", "--lora",
                "--lora_r", "--lora_alpha", "--zo_dpga_clip_threshold",
                "--lr_scheduler_type", "--warmup_ratio", "--weight_decay",
            ]
            if any(flag not in command for flag in required):
                raise AssertionError(f"{job['experiment_id']}: incomplete command {command}")
            if command[command.index("--gradient_accumulation_steps") + 1] != "1":
                raise AssertionError("ZODPGA matrix must keep gradient accumulation at 1")
    stage3 = expand_stage(config, selection, "stage3")
    sample_budgets = {
        job["common"]["batch_size"] * job["common"]["max_steps"] for job in stage3
    }
    if sample_budgets != {40000}:
        raise AssertionError(f"stage3 budgets are not matched: {sample_budgets}")
    if any(job["common"]["dp_epsilon"] != 6.0 for stage in expected for job in expand_stage(config, selection, stage)):
        raise AssertionError("Every matrix job must use nominal epsilon 6")
    print(f"PASS LoRA ZODPGA matrix: {len(all_ids)} unique jobs across four stages")


if __name__ == "__main__":
    main()
