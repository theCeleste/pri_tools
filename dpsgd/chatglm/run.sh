#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/chatglm3-6b [training arguments]"
    echo "Example: bash run.sh /path/to/chatglm3-6b --epochs 4 --max_physical_batch_size 16"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

model_dir="$(cd -- "$1" && pwd)"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${PYTHON_BIN:-python}" dpsgd_chatglm3_subj.py --model_name "$model_dir" "$@"
