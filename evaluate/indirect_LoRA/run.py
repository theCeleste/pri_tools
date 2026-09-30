"""Run the fixed 1200+1200 LoRA neighbourhood attack or evaluate saved scores."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
FPRS = ("0.01", "0.001", "0.0001")
MODELS = {"fm9g": ("9G7B_MHA", "FM9G_MODEL_DIR"),
          "chatglm3": ("chatglm3-6b", "CHATGLM_MODEL_DIR")}
DATASETS = {"sst2": ("sentence", "sentence"),
            "wikitext2": ("text", "wikitext-text")}


def package_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def model_path(model):
    folder, variable = MODELS[model]
    return package_path(os.environ.get(variable, str(ROOT.parent / "models" / folder)))


def attack_command(model, dataset, split, output, *, adapter=None, neighbor_model=None,
                   dataset_file=None, dataset_dir=None, tokenizer=None):
    column, template = DATASETS[dataset]
    adapter = package_path(adapter) if adapter else ROOT / "weights" / model / dataset
    stem = "member" if split == "train" else "nonmember"
    command = [sys.executable, "-u", str(ROOT / "code/attack.py"),
            "--target-model-path", str(model_path(model)),
            "--adapter-path", str(adapter), "--target-tokenizer-path", str(package_path(tokenizer) if tokenizer else adapter),
            "--neighbor-model-path", str(package_path(neighbor_model) if neighbor_model else ROOT / "resources/roberta-base"),
            "--target-dtype", "bfloat16", "--target-device-map", "auto",
            "--target-device", "cuda:0", "--neighbor-device", "cuda:0",
            "--split", split,
            "--text-column", column, "--label-column", "label",
            "--text-template", template, "--max-length", "256",
            "--neighbor-top-k", "8", "--max-neighbors", "100",
            "--score-batch-size", "8", "--num-samples", "1200",
            "--checkpoint-every", "20", "--local-files-only", "--no-resume",
            "--output-pkl", str(output / f"{stem}_scores.pkl"),
            "--metadata-json", str(output / f"{stem}_metadata.json")]
    command += (["--dataset-file", str(package_path(dataset_file))] if dataset_file else
                ["--dataset-dir", str(package_path(dataset_dir) if dataset_dir else ROOT / "data" / dataset)])
    return command


def evaluation_command(scores, output):
    return [sys.executable, "-u", str(ROOT / "code/evaluate.py"),
            "--member-pkl", str(scores / "member_scores.pkl"),
            "--nonmember-pkl", str(scores / "nonmember_scores.pkl"),
            "--target-fprs", *FPRS,
            "--output-csv", str(output / "mia_evaluation_results.csv")]


def validate_scores(directory):
    sys.path.insert(0, str(ROOT / "code"))
    from evaluate import analyze_single_text, first_scalar, load_scores

    for stem in ("member", "nonmember"):
        name = f"{stem}_scores.pkl"
        records = load_scores(str(directory / name))
        if len(records) != 1200:
            raise ValueError(f"{name}: expected 1200 records, got {len(records)}")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or analyze_single_text(record) is None:
                raise ValueError(f"{name} row {index}: missing original or neighbour score")
            values = [first_scalar(value) for value in record.values()]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(f"{name} row {index}: non-finite score")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--mode", choices=("attack", "evaluate", "check"), default="attack")
    parser.add_argument("--gpu", type=int, choices=range(4), default=0,
                        help="Physical GPU index for attack")
    parser.add_argument("--output", help="New output directory; relative paths start in this package")
    parser.add_argument("--scores-dir", help="Directory with member_scores.pkl and nonmember_scores.pkl")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--neighbor-model-path")
    parser.add_argument("--member-data-file")
    parser.add_argument("--nonmember-data-file")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.member_data_file) != bool(args.nonmember_data_file):
        parser.error("--member-data-file and --nonmember-data-file must be provided together")
    if args.dataset_dir and args.member_data_file:
        parser.error("--dataset-dir cannot be combined with data files")
    external = (args.adapter_dir, args.tokenizer_dir, args.neighbor_model_path,
                args.member_data_file, args.nonmember_data_file, args.dataset_dir)
    if args.mode != "attack" and any(external):
        parser.error("external model/data options are only valid for attack")

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONHASHSEED="42", SEED="42",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if args.mode == "check":
        command = [sys.executable, str(ROOT / "verify.py")]
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        return

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required for attack/evaluate")
    if args.mode == "evaluate" and not args.scores_dir:
        parser.error("evaluate requires --scores-dir")
    if args.mode == "attack" and args.scores_dir:
        parser.error("--scores-dir is only valid for evaluate")
    output = package_path(args.output or
                          f"runs/{args.model}_{args.dataset}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    scores = package_path(args.scores_dir) if args.scores_dir else output
    stages = []
    if args.mode == "attack":
        base = model_path(args.model)
        adapter = package_path(args.adapter_dir) if args.adapter_dir else ROOT / "weights" / args.model / args.dataset
        if not args.dry_run and not (base / "config.json").is_file():
            parser.error(f"base model missing: {base}; set {MODELS[args.model][1]}")
        if not args.dry_run and not (adapter / "adapter_model.safetensors").is_file():
            parser.error(f"LoRA weights missing: {adapter}")
        for value in (args.member_data_file, args.nonmember_data_file):
            if value and not args.dry_run and not package_path(value).is_file():
                parser.error(f"dataset file missing: {package_path(value)}")
        stages.extend((f"attack_{stem}", attack_command(
            args.model, args.dataset, split, output, adapter=adapter,
            tokenizer=args.tokenizer_dir, neighbor_model=args.neighbor_model_path,
            dataset_dir=args.dataset_dir,
            dataset_file=args.member_data_file if split == "train" else args.nonmember_data_file))
                      for stem, split in (("member", "train"), ("nonmember", "nonmember")))
    else:
        for stem in ("member", "nonmember"):
            if not (scores / f"{stem}_scores.pkl").is_file():
                parser.error(f"score file missing: {scores / f'{stem}_scores.pkl'}")
    stages.append(("evaluate", evaluation_command(scores, output)))
    print(f"model={args.model} dataset={args.dataset} mode={args.mode} "
          f"physical_gpu={args.gpu} output={output}", flush=True)
    if args.dry_run:
        print("Command preview only; resources have not been validated.")
        for stage, command in stages:
            print(f"{stage}: {shlex.join(command)}")
        return

    output.mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps(
        dict(model=args.model, dataset=args.dataset, mode=args.mode, gpu=args.gpu,
             seed=42, scores=str(scores), commands=dict(stages)), indent=2) + "\n")
    timings = {}
    for stage, command in stages:
        if stage == "evaluate":
            validate_scores(scores)
        log = output / f"{stage}.log"
        print(f"START {stage}; log: {log}", flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            returncode = process.wait()
        timings[stage] = dict(seconds=time.monotonic() - start, returncode=returncode)
        (output / "stage_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        if returncode:
            raise RuntimeError(f"{stage} failed (exit {returncode}); see {log}")
        print(f"DONE {stage}: {timings[stage]['seconds']:.1f}s", flush=True)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
