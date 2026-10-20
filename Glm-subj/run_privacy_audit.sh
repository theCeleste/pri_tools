#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
BASE_MODEL=${BASE_MODEL:-/home/wym/projects/dpfinetune/chatglm3-6b}
RAW_DATA=${RAW_DATA:-$PROJECT_ROOT/data/subj_raw_prepared/subj/none_a3c561096c597e10}
EMBEDDING_DATA=${EMBEDDING_DATA:-$PROJECT_ROOT/data/subj_embedding_protected}
MEMORIZATION_DATA=${MEMORIZATION_DATA:-$PROJECT_ROOT/data/subj_memorization_canary}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_ROOT/outputs/models}
ATTACK_ROOT=${ATTACK_ROOT:-$PROJECT_ROOT/outputs/attacks}
EXAMPLE_ADAPTER=${EXAMPLE_ADAPTER:-$MODEL_ROOT/raw}

GPU_ID=${GPU_ID:-0}
DTYPE=${DTYPE:-bfloat16}
# TRAIN_SAMPLES remains a backwards-compatible override for both method families.
# LoRA uses the complete train split by default; zero means "all".
LORA_TRAIN_SAMPLES=${LORA_TRAIN_SAMPLES:-${TRAIN_SAMPLES:-0}}
ZO_TRAIN_SAMPLES=${ZO_TRAIN_SAMPLES:-${TRAIN_SAMPLES:-0}}
LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-256}
LORA_BATCH_SIZE=${LORA_BATCH_SIZE:-1}
LORA_GRADIENT_ACCUMULATION=${LORA_GRADIENT_ACCUMULATION:-1}
LORA_NO_GRADIENT_CHECKPOINTING=${LORA_NO_GRADIENT_CHECKPOINTING:-0}
EPOCHS=${EPOCHS:-15}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
MAX_LENGTH=${MAX_LENGTH:-512}

ATTACK_TARGETS=${ATTACK_TARGETS:-100}
SAMPLES_PER_PREFIX=${SAMPLES_PER_PREFIX:-1}
PREFIX_TOKENS=${PREFIX_TOKENS:-2}
MIN_SUFFIX_TOKENS=${MIN_SUFFIX_TOKENS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
FUZZY_THRESHOLD=${FUZZY_THRESHOLD:-0.65}
TEMPERATURE_START=${TEMPERATURE_START:-1.20}
TEMPERATURE_END=${TEMPERATURE_END:-1.00}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-50}

# Natural Example experiments use the complete fixed Subj training split.
# Canary commands pass their own diagnostic construction size.
EXAMPLE_SAMPLES=${EXAMPLE_SAMPLES:-7200}
EXAMPLE_CLIP=${EXAMPLE_CLIP:-18.0}
EXAMPLE_NOISE=${EXAMPLE_NOISE:-0.005}
STEERING_MULTIPLIER=${STEERING_MULTIPLIER:-128.0}
EXAMPLE_PROMPT_MODE=${EXAMPLE_PROMPT_MODE:-auto}

ZO_STEPS=${ZO_STEPS:-6750}
ZO_BATCH_SIZE=${ZO_BATCH_SIZE:-16}
ZO_LEARNING_RATE=${ZO_LEARNING_RATE:-5e-5}
ZO_EPSILON=${ZO_EPSILON:-1e-3}
ZO_EVAL_STEPS=${ZO_EVAL_STEPS:-450}
ZO_EVAL_SAMPLES=${ZO_EVAL_SAMPLES:-64}
ZO_LORA_RANK=${ZO_LORA_RANK:-64}
ZO_LORA_ALPHA=${ZO_LORA_ALPHA:-128}
ZO_TARGET_MODULES=${ZO_TARGET_MODULES:-query_key_value}
ZO_TRAINABLE_DTYPE=${ZO_TRAINABLE_DTYPE:-float32}
ZO_LAYER_INDICES=${ZO_LAYER_INDICES:-}
ZO_LOSS_PREFIX_TOKENS=${ZO_LOSS_PREFIX_TOKENS:-8}
ZO_EXPOSURE_SAMPLES=${ZO_EXPOSURE_SAMPLES:-0}
ZO_EXPOSURE_REPEATS=${ZO_EXPOSURE_REPEATS:-1}
ZO_EXPOSURE_MIN_SUFFIX_TOKENS=${ZO_EXPOSURE_MIN_SUFFIX_TOKENS:-8}
ZO_EARLY_STOPPING_PATIENCE=${ZO_EARLY_STOPPING_PATIENCE:-0}
ZO_EARLY_STOPPING_MIN_DELTA=${ZO_EARLY_STOPPING_MIN_DELTA:-0.0}
ZO_AUDIT_DATASET=${ZO_AUDIT_DATASET:-audit_data}
ZO_CHECKPOINT_STEPS=${ZO_CHECKPOINT_STEPS:-450}
ZO_INITIAL_ADAPTER=${ZO_INITIAL_ADAPTER:-}
ZO_ACCOUNTING_PREVIOUS_STEPS=${ZO_ACCOUNTING_PREVIOUS_STEPS:-0}
DP_EPSILON=${DP_EPSILON:-8.0}
DP_DELTA=${DP_DELTA:-1e-5}
DPZERO_CLIP=${DPZERO_CLIP:-30.0}

train_lora() {
  local data_dir=$1 output_dir=$2 original_dir=${3:-}
  local original_args=()
  local checkpoint_args=()
  if [[ -n "$original_dir" ]]; then
    original_args=(--original-data-dir "$original_dir")
  fi
  case "$LORA_NO_GRADIENT_CHECKPOINTING" in
    0) ;;
    1) checkpoint_args=(--no-gradient-checkpointing) ;;
    *) echo "LORA_NO_GRADIENT_CHECKPOINTING must be 0 or 1" >&2; return 2 ;;
  esac
  "$PYTHON_BIN" -m privacy_attack.train_lora \
    --base-model "$BASE_MODEL" --data-dir "$data_dir" --output-dir "$output_dir" \
    --max-train-samples "$LORA_TRAIN_SAMPLES" --rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --batch-size "$LORA_BATCH_SIZE" --gradient-accumulation "$LORA_GRADIENT_ACCUMULATION" \
    --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" --max-grad-norm "$MAX_GRAD_NORM" \
    --max-length "$MAX_LENGTH" --dtype "$DTYPE" --gpu-id "$GPU_ID" \
    "${checkpoint_args[@]}" "${original_args[@]}"
}

build_example_steering() {
  "$PYTHON_BIN" -m privacy_attack.build_steering \
    --base-model "$BASE_MODEL" --data-dir "$RAW_DATA" --output-dir "$MODEL_ROOT/example" \
    --construction-samples "$EXAMPLE_SAMPLES" --clip "$EXAMPLE_CLIP" \
    --noise-multiplier "$EXAMPLE_NOISE" --multiplier "$STEERING_MULTIPLIER" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

build_private_steering() {
  "$PYTHON_BIN" -m privacy_attack.build_steering \
    --base-model "$BASE_MODEL" --data-dir "$RAW_DATA" --output-dir "$MODEL_ROOT/example" \
    --variants private --construction-samples "$EXAMPLE_SAMPLES" --clip "$EXAMPLE_CLIP" \
    --noise-multiplier "$EXAMPLE_NOISE" --multiplier "$STEERING_MULTIPLIER" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

train_zero_order() {
  local method=$1 output_dir=$2
  local layer_args=()
  local initial_args=()
  if [[ -n "$ZO_LAYER_INDICES" ]]; then
    layer_args=(--layers-to-transform "$ZO_LAYER_INDICES")
  fi
  if [[ -n "$ZO_INITIAL_ADAPTER" ]]; then
    initial_args=(--initial-adapter "$ZO_INITIAL_ADAPTER")
  fi
  "$PYTHON_BIN" -m privacy_attack.train_zo \
    --method "$method" --base-model "$BASE_MODEL" --data-dir "$RAW_DATA" --output-dir "$output_dir" \
    --max-train-samples "$ZO_TRAIN_SAMPLES" --rank "$ZO_LORA_RANK" --lora-alpha "$ZO_LORA_ALPHA" \
    --max-steps "$ZO_STEPS" --batch-size "$ZO_BATCH_SIZE" --learning-rate "$ZO_LEARNING_RATE" \
    --zo-epsilon "$ZO_EPSILON" --eval-steps "$ZO_EVAL_STEPS" --eval-samples "$ZO_EVAL_SAMPLES" \
    --max-length "$MAX_LENGTH" \
    --target-modules "$ZO_TARGET_MODULES" --trainable-dtype "$ZO_TRAINABLE_DTYPE" \
    --loss-prefix-tokens "$ZO_LOSS_PREFIX_TOKENS" "${layer_args[@]}" \
    --exposure-samples "$ZO_EXPOSURE_SAMPLES" --exposure-repeats "$ZO_EXPOSURE_REPEATS" \
    --exposure-min-suffix-tokens "$ZO_EXPOSURE_MIN_SUFFIX_TOKENS" \
    --early-stopping-patience "$ZO_EARLY_STOPPING_PATIENCE" \
    --early-stopping-min-delta "$ZO_EARLY_STOPPING_MIN_DELTA" \
    --checkpoint-steps "$ZO_CHECKPOINT_STEPS" \
    "${initial_args[@]}" \
    --accounting-previous-steps "$ZO_ACCOUNTING_PREVIOUS_STEPS" \
    --dp-epsilon "$DP_EPSILON" --dp-delta "$DP_DELTA" --clip-threshold "$DPZERO_CLIP" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

prefix_adapter() {
  local adapter=$1 data_dir=$2 output_dir=$3
  "$PYTHON_BIN" -m privacy_attack.prefix_attack \
    --base-model "$BASE_MODEL" --adapter "$adapter" --data-dir "$data_dir" \
    --output-dir "$output_dir/prefix" --targets-per-split "$ATTACK_TARGETS" \
    --samples-per-prefix "$SAMPLES_PER_PREFIX" --prefix-tokens "$PREFIX_TOKENS" \
    --min-suffix-tokens "$MIN_SUFFIX_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" --fuzzy-threshold "$FUZZY_THRESHOLD" \
    --temperature-start "$TEMPERATURE_START" --temperature-end "$TEMPERATURE_END" \
    --top-p "$TOP_P" --top-k "$TOP_K" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

attack_adapter() {
  local adapter=$1 data_dir=$2 output_dir=$3
  prefix_adapter "$adapter" "$data_dir" "$output_dir"
  "$PYTHON_BIN" -m privacy_attack.causal_mia \
    --base-model "$BASE_MODEL" --adapter "$adapter" --data-dir "$data_dir" \
    --output-dir "$output_dir/mia" --targets-per-split "$ATTACK_TARGETS" \
    --max-length "$MAX_LENGTH" --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

prefix_steering() {
  local steering=$1 data_dir=$2 output_dir=$3
  "$PYTHON_BIN" -m privacy_attack.prefix_attack \
    --base-model "$BASE_MODEL" --adapter "$EXAMPLE_ADAPTER" \
    --steering "$steering" --data-dir "$data_dir" \
    --output-dir "$output_dir/prefix" --targets-per-split "$ATTACK_TARGETS" \
    --samples-per-prefix "$SAMPLES_PER_PREFIX" --prefix-tokens "$PREFIX_TOKENS" \
    --min-suffix-tokens "$MIN_SUFFIX_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" --fuzzy-threshold "$FUZZY_THRESHOLD" \
    --temperature-start "$TEMPERATURE_START" --temperature-end "$TEMPERATURE_END" \
    --top-p "$TOP_P" --top-k "$TOP_K" \
    --steering-multiplier "$STEERING_MULTIPLIER" --prompt-mode "$EXAMPLE_PROMPT_MODE" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

prepare_memorization_data() {
  "$PYTHON_BIN" -m privacy_attack.prepare_memorization \
    --data-dir "$RAW_DATA" --output-dir "$MEMORIZATION_DATA" \
    --records-per-split "${CANARY_RECORDS:-64}"
}

build_example_canary() {
  local canary_multiplier=${CANARY_STEERING_MULTIPLIER:-128}
  local canary_noise=${CANARY_EXAMPLE_NOISE:-0.02}
  "$PYTHON_BIN" -m privacy_attack.build_steering \
    --base-model "$BASE_MODEL" --data-dir "$MEMORIZATION_DATA" \
    --output-dir "$MODEL_ROOT/canary/example" \
    --construction-samples "${CANARY_RECORDS:-64}" --clip "$EXAMPLE_CLIP" \
    --noise-multiplier "$canary_noise" --multiplier "$canary_multiplier" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

train_gradient_canary() {
  local method=$1 output_dir=$2 initial_adapter=${3:-}
  local initial_args=()
  if [[ -n "$initial_adapter" ]]; then
    initial_args=(--initial-adapter "$initial_adapter")
  fi
  "$PYTHON_BIN" -m privacy_attack.train_zo \
    --method "$method" --base-model "$BASE_MODEL" --data-dir "$MEMORIZATION_DATA" \
    --output-dir "$output_dir" --max-train-samples 0 \
    --rank "${CANARY_ZO_RANK:-1}" --lora-alpha "${CANARY_ZO_ALPHA:-2}" \
    --max-steps "${CANARY_ZO_STEPS:-300}" --batch-size "${CANARY_ZO_BATCH_SIZE:-64}" \
    --learning-rate "${CANARY_ZO_LEARNING_RATE:-0.01}" \
    --zo-epsilon "${CANARY_ZO_EPSILON:-0.05}" \
    --eval-steps "${CANARY_ZO_EVAL_STEPS:-50}" --eval-samples "${CANARY_RECORDS:-64}" \
    --checkpoint-steps "${CANARY_ZO_CHECKPOINT_STEPS:-50}" \
    --max-length "${CANARY_MAX_LENGTH:-96}" --target-modules query_key_value \
    --layers-to-transform "${CANARY_ZO_LAYER:-27}" --trainable-dtype "$ZO_TRAINABLE_DTYPE" \
    --loss-prefix-tokens 0 --exposure-samples 0 --exposure-repeats 1 \
    --exposure-min-suffix-tokens 8 --early-stopping-patience "${CANARY_EARLY_STOPPING_PATIENCE:-0}" \
    --early-stopping-min-delta "${CANARY_EARLY_STOPPING_MIN_DELTA:-0.02}" \
    --dp-epsilon "$DP_EPSILON" --dp-delta "$DP_DELTA" \
    --clip-threshold "${CANARY_DPZERO_CLIP:-0.1}" \
    --accounting-previous-steps "$ZO_ACCOUNTING_PREVIOUS_STEPS" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID" "${initial_args[@]}"
}

prefix_canary_adapter() {
  local adapter=$1 output_dir=$2
  "$PYTHON_BIN" -m privacy_attack.prefix_attack \
    --base-model "$BASE_MODEL" --adapter "$adapter" --data-dir "$MEMORIZATION_DATA" \
    --output-dir "$output_dir/prefix" --targets-per-split "${CANARY_ATTACK_TARGETS:-64}" \
    --samples-per-prefix "$SAMPLES_PER_PREFIX" --prefix-tokens 1 --min-suffix-tokens 4 \
    --max-new-tokens "${CANARY_MAX_NEW_TOKENS:-24}" --fuzzy-threshold "$FUZZY_THRESHOLD" \
    --temperature-start "${CANARY_TEMPERATURE_START:-1}" \
    --temperature-end "${CANARY_TEMPERATURE_END:-1}" \
    --top-p "${CANARY_TOP_P:-1}" --top-k "${CANARY_TOP_K:-1}" --prompt-mode raw \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

prefix_canary_steering() {
  local steering=$1 output_dir=$2
  local canary_multiplier=${CANARY_STEERING_MULTIPLIER:-128}
  "$PYTHON_BIN" -m privacy_attack.prefix_attack \
    --base-model "$BASE_MODEL" --steering "$steering" --data-dir "$MEMORIZATION_DATA" \
    --output-dir "$output_dir/prefix" --targets-per-split "${CANARY_ATTACK_TARGETS:-64}" \
    --samples-per-prefix "$SAMPLES_PER_PREFIX" --prefix-tokens 1 --min-suffix-tokens 4 \
    --max-new-tokens "${CANARY_MAX_NEW_TOKENS:-24}" --fuzzy-threshold "$FUZZY_THRESHOLD" \
    --temperature-start "${CANARY_TEMPERATURE_START:-1}" \
    --temperature-end "${CANARY_TEMPERATURE_END:-1}" \
    --top-p "${CANARY_TOP_P:-1}" --top-k "${CANARY_TOP_K:-1}" --prompt-mode raw \
    --steering-multiplier "$canary_multiplier" --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

attack_steering() {
  local steering=$1 data_dir=$2 output_dir=$3
  prefix_steering "$steering" "$data_dir" "$output_dir"
  "$PYTHON_BIN" -m privacy_attack.causal_mia \
    --base-model "$BASE_MODEL" --adapter "$EXAMPLE_ADAPTER" \
    --steering "$steering" --data-dir "$data_dir" \
    --output-dir "$output_dir/mia" --targets-per-split "$ATTACK_TARGETS" \
    --max-length "$MAX_LENGTH" --steering-multiplier "$STEERING_MULTIPLIER" \
    --dtype "$DTYPE" --gpu-id "$GPU_ID"
}

compare_results() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --result embedding-unprotected trained-text "$ATTACK_ROOT/raw/prefix/summary.json" "$ATTACK_ROOT/raw/mia/summary.json" \
    --result embedding-protected trained-text "$ATTACK_ROOT/embedding/known/prefix/summary.json" "$ATTACK_ROOT/embedding/known/mia/summary.json" \
    --result embedding-protected source-original "$ATTACK_ROOT/embedding/original/prefix/summary.json" "$ATTACK_ROOT/embedding/original/mia/summary.json" \
    --result example-mean source-original "$ATTACK_ROOT/example/mean/prefix/summary.json" "$ATTACK_ROOT/example/mean/mia/summary.json" \
    --result example-private source-original "$ATTACK_ROOT/example/private/prefix/summary.json" "$ATTACK_ROOT/example/private/mia/summary.json" \
    --result gradient-mezo trained-text "$ATTACK_ROOT/gradient/mezo/prefix/summary.json" "$ATTACK_ROOT/gradient/mezo/mia/summary.json" \
    --result gradient-dpzero trained-text "$ATTACK_ROOT/gradient/dpzero/prefix/summary.json" "$ATTACK_ROOT/gradient/dpzero/mia/summary.json" \
    --output-json "$PROJECT_ROOT/outputs/attack_comparison.json" \
    --output-csv "$PROJECT_ROOT/outputs/attack_comparison.csv"
}

compare_embedding_direct() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --require-matched-targets \
    --result embedding-raw source-original "$ATTACK_ROOT/raw/prefix/summary.json" \
    --result embedding-cliptext source-original "$ATTACK_ROOT/embedding/original/prefix/summary.json" \
    --output-json "$ATTACK_ROOT/embedding_original_comparison.json" \
    --output-csv "$ATTACK_ROOT/embedding_original_comparison.csv"
}

compare_gradient_direct() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --require-matched-targets \
    --result gradient-mezo trained-text "$ATTACK_ROOT/gradient/mezo/prefix/summary.json" \
    --result gradient-dpzero trained-text "$ATTACK_ROOT/gradient/dpzero/prefix/summary.json" \
    --output-json "$ATTACK_ROOT/gradient_direct_comparison.json" \
    --output-csv "$ATTACK_ROOT/gradient_direct_comparison.csv"
}

compare_example_direct() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --require-matched-targets \
    --result example-mean source-original "$ATTACK_ROOT/example/mean/prefix/summary.json" \
    --result example-private source-original "$ATTACK_ROOT/example/private/prefix/summary.json" \
    --output-json "$ATTACK_ROOT/example_direct_comparison.json" \
    --output-csv "$ATTACK_ROOT/example_direct_comparison.csv"
}

compare_example_canary_direct() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --require-matched-targets \
    --result example-mean canary "$ATTACK_ROOT/canary/example/mean/prefix/summary.json" \
    --result example-private canary "$ATTACK_ROOT/canary/example/private/prefix/summary.json" \
    --output-json "$ATTACK_ROOT/canary/example_direct_comparison.json" \
    --output-csv "$ATTACK_ROOT/canary/example_direct_comparison.csv"
}

compare_gradient_canary_direct() {
  "$PYTHON_BIN" -m privacy_attack.compare \
    --require-matched-targets \
    --result gradient-mezo canary "$ATTACK_ROOT/canary/gradient/mezo/prefix/summary.json" \
    --result gradient-dpzero canary "$ATTACK_ROOT/canary/gradient/dpzero/prefix/summary.json" \
    --output-json "$ATTACK_ROOT/canary/gradient_direct_comparison.json" \
    --output-csv "$ATTACK_ROOT/canary/gradient_direct_comparison.csv"
}

case "${1:-}" in
  prepare-memorization-data) prepare_memorization_data ;;
  train-raw) train_lora "$RAW_DATA" "$MODEL_ROOT/raw" ;;
  train-embedding) train_lora "$EMBEDDING_DATA" "$MODEL_ROOT/embedding" "$RAW_DATA" ;;
  train-embedding-comparison)
    train_lora "$RAW_DATA" "$MODEL_ROOT/raw"
    train_lora "$EMBEDDING_DATA" "$MODEL_ROOT/embedding" "$RAW_DATA"
    ;;
  build-example) build_example_steering ;;
  train-example)
    train_lora "$RAW_DATA" "$MODEL_ROOT/raw"
    build_example_steering
    ;;
  build-example-private) build_private_steering ;;
  build-example-canary) build_example_canary ;;
  train-gradient-mezo) train_zero_order mezo "$MODEL_ROOT/gradient/mezo" ;;
  train-gradient-dpzero) train_zero_order dpzero "$MODEL_ROOT/gradient/dpzero" ;;
  train-gradient-canary-mezo)
    train_gradient_canary mezo "$MODEL_ROOT/canary/gradient/mezo" "${CANARY_MEZO_INITIAL_ADAPTER:-}"
    ;;
  train-gradient-canary-dpzero)
    train_gradient_canary dpzero "$MODEL_ROOT/canary/gradient/dpzero" "${CANARY_DPZERO_INITIAL_ADAPTER:-}"
    ;;
  train-gradient)
    train_zero_order mezo "$MODEL_ROOT/gradient/mezo"
    train_zero_order dpzero "$MODEL_ROOT/gradient/dpzero"
    ;;
  train-protected)
    train_lora "$RAW_DATA" "$MODEL_ROOT/raw"
    train_lora "$EMBEDDING_DATA" "$MODEL_ROOT/embedding" "$RAW_DATA"
    build_private_steering
    train_zero_order dpzero "$MODEL_ROOT/gradient/dpzero"
    ;;
  attack-raw) attack_adapter "$MODEL_ROOT/raw" "$MODEL_ROOT/raw/audit_data" "$ATTACK_ROOT/raw" ;;
  attack-embedding-known) attack_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_data" "$ATTACK_ROOT/embedding/known" ;;
  attack-embedding-original) attack_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_original_data" "$ATTACK_ROOT/embedding/original" ;;
  attack-example-mean) attack_steering "$MODEL_ROOT/example/mean" "$MODEL_ROOT/example/mean/audit_data" "$ATTACK_ROOT/example/mean" ;;
  attack-example-private) attack_steering "$MODEL_ROOT/example/private" "$MODEL_ROOT/example/private/audit_data" "$ATTACK_ROOT/example/private" ;;
  attack-gradient-mezo) attack_adapter "$MODEL_ROOT/gradient/mezo" "$MODEL_ROOT/gradient/mezo/$ZO_AUDIT_DATASET" "$ATTACK_ROOT/gradient/mezo" ;;
  attack-gradient-dpzero) attack_adapter "$MODEL_ROOT/gradient/dpzero" "$MODEL_ROOT/gradient/dpzero/$ZO_AUDIT_DATASET" "$ATTACK_ROOT/gradient/dpzero" ;;
  attack-embedding-known-direct) prefix_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_data" "$ATTACK_ROOT/embedding/known" ;;
  attack-embedding-original-direct) prefix_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_original_data" "$ATTACK_ROOT/embedding/original" ;;
  attack-raw-direct) prefix_adapter "$MODEL_ROOT/raw" "$MODEL_ROOT/raw/audit_data" "$ATTACK_ROOT/raw" ;;
  attack-embedding-comparison-direct)
    prefix_adapter "$MODEL_ROOT/raw" "$MODEL_ROOT/raw/audit_data" "$ATTACK_ROOT/raw"
    prefix_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_original_data" "$ATTACK_ROOT/embedding/original"
    compare_embedding_direct
    ;;
  attack-example-private-direct) prefix_steering "$MODEL_ROOT/example/private" "$MODEL_ROOT/example/private/audit_data" "$ATTACK_ROOT/example/private" ;;
  attack-example-mean-direct) prefix_steering "$MODEL_ROOT/example/mean" "$MODEL_ROOT/example/mean/audit_data" "$ATTACK_ROOT/example/mean" ;;
  attack-example-canary-mean-direct) prefix_canary_steering "$MODEL_ROOT/canary/example/mean" "$ATTACK_ROOT/canary/example/mean" ;;
  attack-example-canary-private-direct) prefix_canary_steering "$MODEL_ROOT/canary/example/private" "$ATTACK_ROOT/canary/example/private" ;;
  attack-gradient-canary-mezo-direct) prefix_canary_adapter "$MODEL_ROOT/canary/gradient/mezo" "$ATTACK_ROOT/canary/gradient/mezo" ;;
  attack-gradient-canary-dpzero-direct) prefix_canary_adapter "$MODEL_ROOT/canary/gradient/dpzero" "$ATTACK_ROOT/canary/gradient/dpzero" ;;
  attack-gradient-mezo-direct) prefix_adapter "$MODEL_ROOT/gradient/mezo" "$MODEL_ROOT/gradient/mezo/$ZO_AUDIT_DATASET" "$ATTACK_ROOT/gradient/mezo" ;;
  attack-gradient-dpzero-direct) prefix_adapter "$MODEL_ROOT/gradient/dpzero" "$MODEL_ROOT/gradient/dpzero/$ZO_AUDIT_DATASET" "$ATTACK_ROOT/gradient/dpzero" ;;
  attack-protected-direct)
    prefix_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_original_data" "$ATTACK_ROOT/embedding/original"
    prefix_steering "$MODEL_ROOT/example/private" "$MODEL_ROOT/example/private/audit_data" "$ATTACK_ROOT/example/private"
    prefix_adapter "$MODEL_ROOT/gradient/dpzero" "$MODEL_ROOT/gradient/dpzero/audit_data" "$ATTACK_ROOT/gradient/dpzero"
    ;;
  train-all)
    train_lora "$RAW_DATA" "$MODEL_ROOT/raw"
    train_lora "$EMBEDDING_DATA" "$MODEL_ROOT/embedding" "$RAW_DATA"
    build_example_steering
    train_zero_order mezo "$MODEL_ROOT/gradient/mezo"
    train_zero_order dpzero "$MODEL_ROOT/gradient/dpzero"
    ;;
  attack-all)
    attack_adapter "$MODEL_ROOT/raw" "$MODEL_ROOT/raw/audit_data" "$ATTACK_ROOT/raw"
    attack_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_data" "$ATTACK_ROOT/embedding/known"
    attack_adapter "$MODEL_ROOT/embedding" "$MODEL_ROOT/embedding/audit_original_data" "$ATTACK_ROOT/embedding/original"
    attack_steering "$MODEL_ROOT/example/mean" "$MODEL_ROOT/example/mean/audit_data" "$ATTACK_ROOT/example/mean"
    attack_steering "$MODEL_ROOT/example/private" "$MODEL_ROOT/example/private/audit_data" "$ATTACK_ROOT/example/private"
    attack_adapter "$MODEL_ROOT/gradient/mezo" "$MODEL_ROOT/gradient/mezo/audit_data" "$ATTACK_ROOT/gradient/mezo"
    attack_adapter "$MODEL_ROOT/gradient/dpzero" "$MODEL_ROOT/gradient/dpzero/audit_data" "$ATTACK_ROOT/gradient/dpzero"
    ;;
  compare) compare_results ;;
  compare-embedding-direct) compare_embedding_direct ;;
  compare-gradient-direct) compare_gradient_direct ;;
  compare-example-direct) compare_example_direct ;;
  compare-example-canary-direct) compare_example_canary_direct ;;
  compare-gradient-canary-direct) compare_gradient_canary_direct ;;
  *)
    echo "Usage: $0 {prepare-memorization-data|build-example-canary|train-gradient-canary-mezo|train-gradient-canary-dpzero|attack-example-canary-mean-direct|attack-example-canary-private-direct|attack-gradient-canary-mezo-direct|attack-gradient-canary-dpzero-direct|compare-example-canary-direct|compare-gradient-canary-direct|train-protected|attack-protected-direct|train-raw|train-embedding|train-embedding-comparison|build-example|train-example|build-example-private|train-gradient-mezo|train-gradient-dpzero|train-gradient|train-all|attack-raw|attack-raw-direct|attack-embedding-known|attack-embedding-original|attack-embedding-comparison-direct|attack-example-mean|attack-example-private|attack-gradient-mezo|attack-gradient-dpzero|attack-embedding-known-direct|attack-embedding-original-direct|attack-example-mean-direct|attack-example-private-direct|attack-gradient-mezo-direct|attack-gradient-dpzero-direct|attack-all|compare|compare-embedding-direct|compare-example-direct|compare-gradient-direct}" >&2
    exit 2
    ;;
esac
