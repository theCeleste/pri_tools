import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import expand_suite, load_config  # noqa: E402


STAGES = {
    "dev_zero_shot_control": {"suite": "dev_zero_shot_control", "seed": None},
    "mezo_zo_eps_sweep": {"suite": "mezo_zo_eps_sweep", "seed": None},
    "head_dp_batch_sweep": {"suite": "head_dp_batch_sweep", "seed": None},
    "mezo_lr_sweep": {"suite": "mezo_lr_sweep", "seed": None},
    "head_dp_clip_sweep": {"suite": "head_dp_clip_sweep", "seed": None},
    "mezo_baseline": {"suite": "mezo_baseline", "seed": None},
    "mezo_calibrated_baseline": {"suite": "mezo_calibrated_baseline", "seed": None},
    "zo_dpga_seed0": {"suite": "zo_dpga_budget_sweep", "seed": 0},
    "zo_dpga_seed1": {"suite": "zo_dpga_budget_sweep", "seed": 1},
    "zo_dpga_seed2": {"suite": "zo_dpga_budget_sweep", "seed": 2},
    "zo_dpga_calibrated_seed0": {"suite": "zo_dpga_calibrated_budget_sweep", "seed": 0},
    "zo_dpga_calibrated_seed1": {"suite": "zo_dpga_calibrated_budget_sweep", "seed": 1},
    "zo_dpga_calibrated_seed2": {"suite": "zo_dpga_calibrated_budget_sweep", "seed": 2},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def selected_jobs(config, stage):
    definition = STAGES[stage]
    jobs = expand_suite(config, definition["suite"])
    return [
        (index, job)
        for index, job in enumerate(jobs)
        if definition["seed"] is None or job["seed"] == definition["seed"]
    ]


def output_path(job):
    return LLAMA_DIR / job["output_dir"]


def completion_state(job):
    output = output_path(job)
    required = [
        output / "run_config.json",
        output / "evaluation_benchmark.json",
        output / "eval_metrics.json",
    ]
    if job["method"] != "zero_shot":
        required.append(output / "benchmark.json")
    if job["method"] == "zo_dpga":
        checkpoints = sorted(output.glob("checkpoint-*/zo_dpga_privacy.json"))
        if not checkpoints:
            required.append(output / "checkpoint-*/zo_dpga_privacy.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return False, missing
    try:
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "eval_metrics.json").read_text(encoding="utf-8"))
        benchmark = (
            json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
            if job["method"] != "zero_shot" else None
        )
    except (json.JSONDecodeError, OSError) as error:
        return False, [f"invalid result JSON: {error}"]
    if config.get("experiment_id") != job["experiment_id"]:
        return False, ["run_config experiment_id mismatch"]
    if benchmark is not None and benchmark.get("experiment_id") != job["experiment_id"]:
        return False, ["benchmark experiment_id mismatch"]
    if not metrics:
        return False, ["empty eval_metrics.json"]
    return True, []


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=LLAMA_DIR / "configs" / "formal_experiments.yaml")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm", help="Must exactly match --stage when executing")
    parser.add_argument("--timeout_hours", type=float, default=72.0)
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = selected_jobs(config, args.stage)
    print(f"Stage {args.stage}: {len(jobs)} sequential jobs")
    for order, (suite_index, job) in enumerate(jobs):
        complete, missing = completion_state(job)
        state = "complete" if complete else "pending"
        print(f"[{order:02d}] suite-index={suite_index:02d} {state:8s} {job['experiment_id']}")
        if missing and output_path(job).exists():
            print(f"     incomplete artifacts: {missing}")

    if not args.run:
        print(f"Dry run only. Execute with --run --confirm {args.stage}")
        return
    if args.confirm != args.stage:
        raise ValueError(f"Execution requires --confirm {args.stage}")
    if args.timeout_hours <= 0:
        raise ValueError("timeout_hours must be positive")

    status_file = LLAMA_DIR / config["output_root"] / "stages" / f"{args.stage}.json"
    status = {
        "format": "zo_dpga-formal-stage-v1",
        "stage": args.stage,
        "started_at": now(),
        "updated_at": now(),
        "status": "running",
        "jobs": [],
    }
    write_status(status_file, status)

    for suite_index, job in jobs:
        complete, _ = completion_state(job)
        if complete:
            status["jobs"].append({"experiment_id": job["experiment_id"], "status": "skipped_complete"})
            status["updated_at"] = now()
            write_status(status_file, status)
            continue

        command = [
            sys.executable,
            "run_formal_experiment.py",
            "--suite", job["suite"],
            "--index", str(suite_index),
            "--run",
        ]
        if output_path(job).exists():
            command.append("--resume")
        entry = {"experiment_id": job["experiment_id"], "status": "running", "started_at": now()}
        status["jobs"].append(entry)
        status["updated_at"] = now()
        write_status(status_file, status)
        try:
            subprocess.run(
                command,
                cwd=LLAMA_DIR,
                check=True,
                timeout=args.timeout_hours * 3600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            entry.update({"status": "failed", "finished_at": now(), "error": str(error)})
            status.update({"status": "failed", "updated_at": now()})
            write_status(status_file, status)
            raise
        complete, missing = completion_state(job)
        if not complete:
            entry.update({"status": "failed_validation", "finished_at": now(), "missing": missing})
            status.update({"status": "failed", "updated_at": now()})
            write_status(status_file, status)
            raise RuntimeError(f"Completed process has invalid artifacts for {job['experiment_id']}: {missing}")
        entry.update({"status": "completed", "finished_at": now()})
        status["updated_at"] = now()
        write_status(status_file, status)

    status.update({"status": "completed", "finished_at": now(), "updated_at": now()})
    write_status(status_file, status)
    print(f"PASS formal stage {args.stage}: {len(jobs)} jobs complete")


if __name__ == "__main__":
    main()
