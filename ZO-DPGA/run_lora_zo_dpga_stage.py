import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_lora_zo_dpga_matrix import (
    DEFAULT_CONFIG,
    DEFAULT_SELECTION,
    LLAMA_DIR,
    expand_stage,
    load_yaml,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def output_path(job):
    return LLAMA_DIR / job["output_dir"]


def completion_state(job):
    output = output_path(job)
    required = [
        output / "run_config.json",
        output / "benchmark.json",
        output / "evaluation_benchmark.json",
        output / "eval_metrics.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if job["method"] == "zo_dpga":
        privacy_manifests = list(output.glob("checkpoint-*/zo_dpga_privacy.json"))
        if not privacy_manifests:
            missing.append(str(output / "checkpoint-*/zo_dpga_privacy.json"))
    if missing:
        return False, missing
    try:
        run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "eval_metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return False, [f"invalid result JSON: {error}"]
    if run_config.get("experiment_id") != job["experiment_id"]:
        return False, ["run_config experiment_id mismatch"]
    if benchmark.get("experiment_id") != job["experiment_id"]:
        return False, ["benchmark experiment_id mismatch"]
    if not metrics:
        return False, ["empty eval_metrics.json"]
    return True, []


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run one parameter-matrix stage sequentially")
    parser.add_argument("--runner-module", default="run_lora_zo_dpga_matrix")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm", help="Must exactly match --stage when executing")
    parser.add_argument("--timeout_hours", type=float, default=72.0)
    args = parser.parse_args()

    runner = importlib.import_module(args.runner_module)
    config = load_yaml(args.config)
    selection = load_yaml(args.selection)
    jobs = runner.expand_stage(config, selection, args.stage)
    print(f"Stage {args.stage}: {len(jobs)} sequential jobs")
    for index, job in enumerate(jobs):
        complete, missing = completion_state(job)
        print(f"[{index:02d}] {'complete' if complete else 'pending':8s} {job['experiment_id']}")
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
        "format": "lora-zo_dpga-parameter-stage-v1",
        "stage": args.stage,
        "started_at": now(),
        "updated_at": now(),
        "status": "running",
        "jobs": [],
    }
    write_status(status_file, status)
    for index, job in enumerate(jobs):
        complete, _ = completion_state(job)
        if complete:
            status["jobs"].append({"experiment_id": job["experiment_id"], "status": "skipped_complete"})
            write_status(status_file, status)
            continue
        command = [
            sys.executable,
            str(LLAMA_DIR / f"{args.runner_module}.py"),
            "--config", str(args.config),
            "--selection", str(args.selection),
            "--stage", args.stage,
            "--index", str(index),
            "--run",
        ]
        if output_path(job).exists():
            command.append("--resume")
        entry = {"experiment_id": job["experiment_id"], "status": "running", "started_at": now()}
        status["jobs"].append(entry)
        status["updated_at"] = now()
        write_status(status_file, status)
        try:
            subprocess.run(command, cwd=LLAMA_DIR, check=True, timeout=args.timeout_hours * 3600)
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
            raise RuntimeError(f"Invalid artifacts for {job['experiment_id']}: {missing}")
        entry.update({"status": "completed", "finished_at": now()})
        status["updated_at"] = now()
        write_status(status_file, status)

    status.update({"status": "completed", "finished_at": now(), "updated_at": now()})
    write_status(status_file, status)
    print(f"PASS {args.stage}: {len(jobs)} jobs complete")


if __name__ == "__main__":
    main()
