#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null || {
    printf 'Python unavailable; activate dp-few-shot or set PYTHON to its interpreter path\n' >&2
    exit 127
}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU=3
MODEL=fm9g
DATASET=sst2
MODE=full
OUTPUT=""
INPUT_JSON=""
RESULTS=""
DRY_RUN=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --input-json) INPUT_JSON="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=(--dry-run); shift ;;
        --help|-h)
            printf 'Usage: bash run.sh [--model fm9g|chatglm3] [--dataset sst2|wikitext2] [--gpu 0|1|2|3] [--mode full|attack|evaluate|check] [--input-json path] [--results path] [--output path] [--dry-run]\n'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$GPU" in 0|1|2|3) ;; *) printf 'GPU must be 0, 1, 2 or 3\n' >&2; exit 2 ;; esac
case "$MODEL" in fm9g|chatglm3) ;; *) printf 'Model must be fm9g or chatglm3\n' >&2; exit 2 ;; esac
case "$DATASET" in sst2|wikitext2) ;; *) printf 'Dataset must be sst2 or wikitext2\n' >&2; exit 2 ;; esac
# Map one physical GPU to logical cuda:0, even if the caller has a visibility setting.
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
if [[ -n "$INPUT_JSON" && "$MODE" != attack ]]; then
    printf '%s\n' '--input-json is only valid for attack mode' >&2; exit 2
fi
if [[ -n "$RESULTS" && "$MODE" != evaluate ]]; then
    printf '%s\n' '--results is only valid for evaluate mode' >&2; exit 2
fi
if [[ "$MODE" == check ]]; then
    "$PYTHON" "$ROOT/verify.py" --model "$MODEL"
    export DELIVERY_TEST_MODEL="$MODEL"
    "$PYTHON" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
    exit 0
fi
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="runs/${MODEL}_${DATASET}_${MODE}_seed0_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "$OUTPUT" != /* ]]; then OUTPUT="$ROOT/$OUTPUT"; fi
if [[ "$DATASET" == wikitext2 ]]; then
    DATA_OPTIONS=(--task-dataset wikitext2 --wikitext2-dir data/WikiText-2-raw-v1
                  --wikitext2-config wikitext-2-raw-v1 --num-test 0
                  --max-length 2048 --proposal-max-length 512)
else
    DATA_OPTIONS=(--task-dataset sst2 --sst2-dir data/SST-2 --num-test 20 --max-length 512)
fi
case "$MODE" in
    full|attack)
        INPUT=()
        if [[ "$MODE" == attack ]]; then
            if [[ -z "$INPUT_JSON" ]]; then
                printf '%s\n' 'attack requires --input-json path/to/synthetic_text_results.json' >&2; exit 2
            fi
            if [[ "$INPUT_JSON" != /* ]]; then INPUT_JSON="$ROOT/$INPUT_JSON"; fi
            [[ -f "$INPUT_JSON" ]] || { printf 'Input JSON not found: %s\n' "$INPUT_JSON" >&2; exit 2; }
            INPUT=(--input-json "$INPUT_JSON")
        fi
        printf 'MODEL=%s dataset=%s mode=%s physical_gpu=%s output=%s\n' "$MODEL" "$DATASET" "$MODE" "$GPU" "$OUTPUT"
        exec "$PYTHON" -u "$ROOT/DPICL/scripts/run_indirect.py" \
            --target-model "$MODEL" --trust-remote-code --gpu 0 \
            --run-dir "$OUTPUT" "${DATA_OPTIONS[@]}" \
            --proposal-model resources/bert-base-uncased \
            --num-nonmembers 500 --num-demonstrations 10 \
            --seed 0 --score-mode token-mean --dtype float32 \
            --generation-dtype float16 --max-token-per-text 100 \
            --prompt-start 5 --prompt-stop 10 "${INPUT[@]}" "${DRY_RUN[@]}"
        ;;
    evaluate)
        if [[ -z "$RESULTS" ]]; then
            printf '%s\n' 'evaluate requires --results path/to/experiment_directory' >&2; exit 2
        fi
        if [[ "$RESULTS" != /* ]]; then RESULTS="$ROOT/$RESULTS"; fi
        if [[ ${#DRY_RUN[@]} -ne 0 ]]; then
            printf 'evaluate does not accept --dry-run\n' >&2; exit 2
        fi
        if [[ -e "$OUTPUT" ]]; then
            printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT" >&2; exit 73
        fi
        for FIELD in original_text synthetic_text; do
            for SOURCE in other "$DATASET"; do
                FILE="$RESULTS/$FIELD/attack_results_${SOURCE}_bert_0_${FIELD}.csv"
                [[ -f "$FILE" ]] || { printf 'Attack CSV not found: %s\n' "$FILE" >&2; exit 2; }
            done
        done
        for FIELD in original_text synthetic_text; do
            "$PYTHON" "$ROOT/DPICL/Indirect_attack/combine.py" \
                --member-file "$RESULTS/$FIELD/attack_results_other_bert_0_${FIELD}.csv" \
                --nonmember-file "$RESULTS/$FIELD/attack_results_${DATASET}_bert_0_${FIELD}.csv" \
                --output "$OUTPUT/$FIELD/combined.csv"
            "$PYTHON" "$ROOT/DPICL/Indirect_attack/evaluate_attack_metrics.py" \
                --file "$OUTPUT/$FIELD/combined.csv" --output-dir "$OUTPUT/$FIELD/evaluation"
        done
        printf 'Evaluation completed: %s\n' "$OUTPUT"
        ;;
    *) printf 'Mode must be full, attack, evaluate or check\n' >&2; exit 2 ;;
esac
