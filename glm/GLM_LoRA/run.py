"""Single-GPU entry point for fixed SST-2 LoRA and neighbourhood MIA settings."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FPRS = ["1.00", "0.50", "0.10", "0.01", "0.001", "0.0001"]


def local_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def base_model_dir():
    return local_path(os.environ.get("CHATGLM_MODEL_DIR", str(ROOT.parent / "models/chatglm3-6b")))


def build_attack(adapter, split, output):
    return [sys.executable, "-u", str(ROOT / "code/attack.py"),
            "--target-model-path", str(base_model_dir()),
            "--adapter-path", str(adapter), "--target-tokenizer-path", str(adapter),
            "--neighbor-model-path", str(ROOT / "resources/roberta-base"),
            "--target-dtype", "bfloat16", "--target-device-map", "auto",
            "--target-device", "cuda:0", "--neighbor-device", "cuda:0",
            "--dataset-dir", str(ROOT / "data"), "--split", split,
            "--text-column", "sentence", "--label-column", "label",
            "--text-template", "sentence", "--max-length", "256",
            "--neighbor-top-k", "8", "--max-neighbors", "100",
            "--score-batch-size", "8", "--num-samples", "1200",
            "--checkpoint-every", "20", "--local-files-only", "--no-resume",
            "--output-pkl", str(output / f"{'member' if split == 'train' else 'nonmember'}_scores.pkl"),
            "--metadata-json", str(output / f"{'member' if split == 'train' else 'nonmember'}_metadata.json")]


def build_evaluation(scores, output):
    return [sys.executable, "-u", str(ROOT / "code/evaluate.py"),
            "--member-pkl", str(scores / "member_scores.pkl"),
            "--nonmember-pkl", str(scores / "nonmember_scores.pkl"),
            "--target-fprs", *FPRS,
            "--output-csv", str(output / "mia_evaluation_results.csv")]


def validate_scores(directory):
    sys.path.insert(0, str(ROOT / "code"))
    from evaluate import analyze_single_text, first_scalar, load_scores
    for name in ("member_scores.pkl", "nonmember_scores.pkl"):
        records = load_scores(str(directory / name))
        if len(records) != 1200:
            raise ValueError(f"Expected 1200 scored samples in {name}, got {len(records)}")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or analyze_single_text(record) is None:
                raise ValueError(f"Missing original score or neighbours: {name} row {index}")
            values = [first_scalar(value) for value in record.values()]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(f"Invalid score: {name} row {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["attack", "full", "train", "evaluate", "check"], default="attack")
    parser.add_argument("--gpu", type=int, choices=range(4), default=3, help="Physical GPU index")
    parser.add_argument("--output", help="New output directory, relative to this package")
    parser.add_argument("--adapter-dir", help="Adapter for attack mode; defaults to weights/lora")
    parser.add_argument("--scores-dir", help="Directory with two score pkls, required for evaluate mode")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    if args.adapter_dir and args.mode != "attack":
        parser.error("--adapter-dir is only valid for attack mode")
    if args.scores_dir and args.mode != "evaluate":
        parser.error("--scores-dir is only valid for evaluate mode")
    if args.mode == "evaluate" and not args.scores_dir:
        parser.error("evaluate requires --scores-dir")

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONHASHSEED="42", SEED="42",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if args.mode == "check":
        for script in ("verify.py", "tests/test_delivery.py"):
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
        return

    output = local_path(args.output or f"runs/{args.mode}_{datetime.now():%Y%m%d_%H%M%S}")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    adapter = local_path(args.adapter_dir or "weights/lora")
    if args.mode in ("train", "full"):
        adapter = output / "adapter"
    if args.mode == "attack" and not (adapter / "adapter_model.safetensors").is_file():
        parser.error(f"Adapter weights not found: {adapter}")
    if args.mode in ("train", "full", "attack") and not (base_model_dir() / "config.json").is_file():
        parser.error(f"External ChatGLM model not found: {base_model_dir()}; set CHATGLM_MODEL_DIR")
    scores = local_path(args.scores_dir) if args.scores_dir else output
    if args.mode == "evaluate":
        for name in ("member_scores.pkl", "nonmember_scores.pkl"):
            if not (scores / name).is_file():
                parser.error(f"Score file not found: {scores / name}")

    env.update(GPU_IDS=str(args.gpu), MODEL_PATH=str(base_model_dir()),
               DATA_DIR=str(ROOT / "data"), OUTPUT_DIR=str(adapter), TASK="sst2",
               NUM_EPOCHS="5", BATCH_SIZE="4", GRADIENT_ACCUMULATION_STEPS="2",
               LEARNING_RATE="2e-4", MAX_LENGTH="256", LORA_R="16", LORA_ALPHA="32",
               LORA_DROPOUT="0.0",
               TARGET_MODULES="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    stages = []
    if args.mode in ("train", "full"):
        stages.append(("train", [sys.executable, "-u", str(ROOT / "code/train.py")]))
    if args.mode in ("attack", "full"):
        stages.extend((f"attack_{split}", build_attack(adapter, split, output))
                      for split in ("train", "nonmember"))
    if args.mode != "train":
        stages.append(("evaluate", build_evaluation(scores, output)))
    print(f"Mode={args.mode} physical_gpu={args.gpu} output={output}", flush=True)
    if args.dry_run:
        import shlex
        for name, command in stages:
            print(f"{name}: {shlex.join(command)}")
        return

    output.mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps(
        dict(mode=args.mode, gpu=args.gpu, seed=42, adapter=str(adapter),
             scores=str(scores), commands=dict(stages)), indent=2) + "\n")
    timings = {}
    for name, command in stages:
        if name == "evaluate":
            validate_scores(scores)
        log = output / f"{name}.log"
        print(f"START {name}; log: {log}", flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        timings[name] = dict(seconds=time.monotonic() - start, returncode=result.returncode)
        (output / "stage_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")
        print(f"DONE {name}: {timings[name]['seconds']:.1f}s", flush=True)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
