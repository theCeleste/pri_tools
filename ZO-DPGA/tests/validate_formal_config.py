import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for, expand_suite, load_config  # noqa: E402


def main():
    config_path = LLAMA_DIR / "configs" / "formal_experiments.yaml"
    config = load_config(config_path)
    expected_counts = {
        "dev_zero_shot_control": 1,
        "mezo_zo_eps_sweep": 6,
        "head_dp_batch_sweep": 2,
        "mezo_lr_sweep": 6,
        "head_dp_clip_sweep": 3,
        "zero_shot_control": 1,
        "mezo_utility_pilot": 3,
        "zo_dpga_utility_pilot": 3,
        "performance_smoke": 3,
        "mezo_baseline": 9,
        "zo_dpga_budget_sweep": 36,
        "mezo_calibrated_baseline": 9,
        "zo_dpga_calibrated_budget_sweep": 36,
    }
    all_ids = set()
    all_outputs = set()
    for suite, expected_count in expected_counts.items():
        jobs = expand_suite(config, suite)
        if len(jobs) != expected_count:
            raise AssertionError(f"{suite}: expected {expected_count} jobs, got {len(jobs)}")
        for job in jobs:
            if job["experiment_id"] in all_ids or str(job["output_dir"]) in all_outputs:
                raise AssertionError(f"Duplicate formal experiment identity: {job}")
            all_ids.add(job["experiment_id"])
            all_outputs.add(str(job["output_dir"]))
            command = command_for(job)
            required = [
                "--experiment_suite",
                "--experiment_id",
                "--result_file",
                "--train_set_seed",
            ]
            if job["method"] != "zero_shot":
                required.append("--load_best_model_at_end")
            if any(flag not in command for flag in required):
                raise AssertionError(f"{job['experiment_id']}: incomplete command {command}")
            if job["method"] == "zo_dpga" and "--zo_dpga" not in command:
                raise AssertionError(f"{job['experiment_id']}: missing ZODPGA flag")
            if job["method"] == "mezo" and "--zo_dpga" in command:
                raise AssertionError(f"{job['experiment_id']}: non-private baseline contains ZODPGA flag")
            if job["method"] == "zero_shot":
                if command[command.index("--trainer") + 1] != "none" or "--max_steps" in command:
                    raise AssertionError(f"{job['experiment_id']}: invalid zero-shot command {command}")
            if job.get("dev_only") and "--dev_only" not in command:
                raise AssertionError(f"{job['experiment_id']}: tuning job can access formal test data")
            if suite == "mezo_calibrated_baseline":
                expected = {
                    "lora": (1.0e-5, 1.0e-3, 4, 10.0),
                    "prefix": (1.0e-5, 1.0e-4, 4, 10.0),
                    "head": (1.0e-7, 1.0e-3, 4, 10.0),
                }[job["mode"]]
                actual = tuple(job["common"][key] for key in (
                    "learning_rate", "zo_eps", "batch_size", "dp_clip",
                ))
                if actual != expected:
                    raise AssertionError(f"{job['experiment_id']}: expected {expected}, got {actual}")
            if suite == "zo_dpga_calibrated_budget_sweep":
                expected = {
                    "lora": (1.0e-5, 1.0e-3, 4, 10.0),
                    "prefix": (1.0e-5, 1.0e-4, 4, 10.0),
                    "head": (1.0e-7, 1.0e-3, 4, 1.0),
                }[job["mode"]]
                actual = tuple(job["common"][key] for key in (
                    "learning_rate", "zo_eps", "batch_size", "dp_clip",
                ))
                if actual != expected:
                    raise AssertionError(f"{job['experiment_id']}: expected {expected}, got {actual}")
    print(f"PASS formal config: {len(all_ids)} unique jobs across {len(expected_counts)} suites")


if __name__ == "__main__":
    main()
