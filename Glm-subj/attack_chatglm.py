#!/usr/bin/env python3
"""Unified command dispatcher for ChatGLM3/Subj privacy experiments."""

from __future__ import annotations

import argparse


COMMANDS = {
    "train-lora": "privacy_attack.train_lora",
    "build-steering": "privacy_attack.build_steering",
    "prepare-synthetic-baseline": "privacy_attack.prepare_example",
    "train-zero-order": "privacy_attack.train_zo",
    "train-dpsgd-baseline": "privacy_attack.train_dp",
    "prefix-attack": "privacy_attack.prefix_attack",
    "causal-mia": "privacy_attack.causal_mia",
    "compare": "privacy_attack.compare",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparsers.add_parser(command, add_help=False)
    args, remainder = parser.parse_known_args(argv)
    if args.command == "train-lora":
        from privacy_attack.train_lora import main as run
    elif args.command == "build-steering":
        from privacy_attack.build_steering import main as run
    elif args.command == "prepare-synthetic-baseline":
        from privacy_attack.prepare_example import main as run
    elif args.command == "train-zero-order":
        from privacy_attack.train_zo import main as run
    elif args.command == "train-dpsgd-baseline":
        from privacy_attack.train_dp import main as run
    elif args.command == "prefix-attack":
        from privacy_attack.prefix_attack import main as run
    elif args.command == "causal-mia":
        from privacy_attack.causal_mia import main as run
    else:
        from privacy_attack.compare import main as run
    return run(remainder)


if __name__ == "__main__":
    main()
