from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "example": ("mean", "private"),
    "embedding": ("raw", "cliptext_lora"),
    "gradient": ("mezo", "dpzero"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SUBJ indirect membership inference")
    parser.add_argument("--mode", choices=("check", "full", "attack", "evaluate"), default="full")
    parser.add_argument("--method", choices=tuple(VARIANTS), default="example")
    parser.add_argument("--variant", default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("CHATGLM_MODEL_DIR", "../models/chatglm3-6b"))
    parser.add_argument("--output")
    parser.add_argument("--neighbors", help="Neighbor JSONL used by attack mode")
    parser.add_argument("--proposal-model", default="resources/bert-base-uncased")
    parser.add_argument("--baseline-adapter")
    parser.add_argument("--protected-adapter")
    parser.add_argument("--splits")
    parser.add_argument("--interface-dir", default="example/interface")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable_command = " ".join(command).replace("dpzero", "zo_dpga")
    print("$ " + printable_command, flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def selected_variants(method: str, variant: str, *, full: bool = False) -> tuple[str, ...]:
    allowed = VARIANTS[method]
    if full or variant == "both":
        return allowed
    if variant not in allowed:
        raise ValueError(f"{method} supports variants: both, {', '.join(allowed)}")
    return (variant,)


def default_output(method: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{method}_{stamp}"


def evaluate(method: str, variants: tuple[str, ...], output: Path, env: dict[str, str], dry_run: bool) -> None:
    counts = {"example": (1000, 1000), "embedding": (100, 1000), "gradient": (100, 1000)}
    members, nonmembers = counts[method]
    for variant in variants:
        score_file = output / variant / "scores.csv"
        evaluation = output / variant / "evaluation"
        run(
            [
                sys.executable,
                "evaluate.py",
                "--scores",
                str(score_file),
                "--output",
                str(evaluation),
                "--members",
                str(members),
                "--nonmembers",
                str(nonmembers),
            ],
            env,
            dry_run,
        )


def run_example(args: argparse.Namespace, output: Path, env: dict[str, str]) -> None:
    variants = selected_variants(args.method, args.variant, full=args.mode == "full")
    variant_arg = "both" if len(variants) == 2 else variants[0]
    full_pool = output / "full_pool"
    command = [
        sys.executable,
        "example/code/run_steering.py",
        "--model",
        args.model,
        "--interface-dir",
        args.interface_dir,
        "--proposal-model",
        args.proposal_model,
        "--output",
        str(full_pool),
        "--device",
        "cuda:0",
        "--member-count",
        "1000",
        "--nonmember-count",
        "2000",
        "--construction-samples",
        "1000",
        "--construction-seed",
        "42",
        "--seed",
        "43",
        "--max-length",
        "512",
        "--proposal-max-length",
        "512",
        "--neighbor-top-k",
        "6",
        "--candidate-neighbors",
        "50",
        "--num-neighbors",
        "10",
        "--score-batch-size",
        "16",
        "--layers",
        "11",
        "12",
        "13",
        "14",
        "15",
        "--read-token-index",
        "-2",
        "--clip",
        "18",
        "--noise-multiplier",
        "0.005",
        "--steering-multiplier",
        "256",
        "--private-seed",
        "42",
        "--variants",
        variant_arg,
    ]
    if args.neighbors:
        command.extend(("--neighbor-cache", args.neighbors))
    if args.resume:
        command.append("--resume")
    run(command, env, args.dry_run)
    run(
        [
            sys.executable,
            "example/code/select_subset.py",
            "--input",
            str(full_pool),
            "--output",
            str(output),
            "--variants",
            *variants,
            "--member-count",
            "1000",
            "--nonmember-pool-count",
            "2000",
            "--nonmember-count",
            "1000",
            "--subset-seed",
            "60",
        ],
        env,
        args.dry_run,
    )
    evaluate(args.method, variants, output, env, args.dry_run)


def prepare_neighbors(method: str, output: Path, env: dict[str, str], dry_run: bool, *, splits=None, proposal_model="resources/bert-base-uncased") -> Path:
    neighbors = output / "neighbors.jsonl"
    run(
        [
            sys.executable,
            f"{method}/code/prepare_neighbors.py",
            "--splits",
            splits or f"{method}/data/splits",
            "--proposal-model",
            proposal_model,
            "--output",
            str(neighbors),
            "--device",
            "cuda:0",
            "--seed",
            "42",
            "--proposal-max-length",
            "512",
            "--dropout",
            "0.7",
            "--neighbor-top-k",
            "6",
            "--candidate-neighbors",
            "50",
            "--num-neighbors",
            "10",
        ],
        env,
        dry_run,
    )
    return neighbors


def run_adapter_method(args: argparse.Namespace, output: Path, env: dict[str, str]) -> None:
    variants = selected_variants(args.method, args.variant, full=args.mode == "full")
    if args.mode == "full":
        neighbors = prepare_neighbors(args.method, output, env, args.dry_run,
                                      splits=args.splits, proposal_model=args.proposal_model)
    else:
        neighbors = Path(args.neighbors) if args.neighbors else output / "neighbors.jsonl"
        if not args.dry_run and not neighbors.is_file():
            raise FileNotFoundError(
                f"Neighbor file not found: {neighbors}. Run full mode first or pass --neighbors."
            )

    settings = {
        "embedding": {"max_length": "512", "prefix": "0"},
        "gradient": {"max_length": "14", "prefix": "8"},
    }[args.method]
    for variant in variants:
        run(
            [
                sys.executable,
                f"{args.method}/code/attack.py",
                "--model",
                args.model,
                "--adapter",
                (args.baseline_adapter if variant == VARIANTS[args.method][0] else args.protected_adapter)
                or f"{args.method}/weights/{variant}",
                "--splits",
                args.splits or f"{args.method}/data/splits",
                "--neighbors",
                str(neighbors),
                "--output",
                str(output / variant),
                "--device",
                "cuda:0",
                "--dtype",
                "bf16",
                "--max-length",
                settings["max_length"],
                "--loss-prefix-tokens",
                settings["prefix"],
                "--score-batch-size",
                "16",
                *( ["--resume"] if args.resume else [] ),
            ],
            env,
            args.dry_run,
        )


def main() -> None:
    args = parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    if args.mode == "check":
        run([sys.executable, "verify.py", "--model", args.model], env, args.dry_run)
        return

    output = Path(args.output) if args.output else default_output(args.method)
    variants = selected_variants(args.method, args.variant, full=args.mode == "full")
    if args.mode == "evaluate":
        evaluate(args.method, variants, output, env, args.dry_run)
        return

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    if args.method == "example":
        run_example(args, output, env)
    else:
        run_adapter_method(args, output, env)


if __name__ == "__main__":
    main()
