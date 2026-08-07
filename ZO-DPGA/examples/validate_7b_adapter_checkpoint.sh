#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
TASK=${TASK:-SST2}
MODE=${MODE:-prefix}
BS=${BS:-1}
LR=${LR:-1e-5}
ZO_EPS=${ZO_EPS:-1e-3}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-result/adapter-checkpoint-validation-${MODE}-${RUN_ID}}

case "$MODE" in
    prefix)
        MODE_ARGS=(--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act)
        ;;
    lora)
        MODE_ARGS=(--lora)
        ;;
    head)
        MODE_ARGS=(--head_tuning)
        ;;
    *)
        echo "MODE must be prefix, lora, or head" >&2
        exit 2
        ;;
esac

COMMON_ARGS=(
    --model_name "$MODEL"
    --task_name "$TASK"
    --output_dir "$OUTPUT_DIR"
    --tag "adapter-checkpoint-validation-$MODE"
    --train_set_seed 0
    --num_train 8
    --num_dev 2
    --num_eval 2
    --trainer zo
    --load_bfloat16
    --learning_rate "$LR"
    --zo_eps "$ZO_EPS"
    --per_device_train_batch_size "$BS"
    --gradient_accumulation_steps 1
    --lr_scheduler_type constant
    --logging_steps 1
    --save_strategy steps
    --save_steps 1
    --save_total_limit 2
    --evaluation_strategy no
    --train_as_classification
    --no_eval
    "${MODE_ARGS[@]}"
)

echo "Validation output: $OUTPUT_DIR"
echo "Stage 1: train and save checkpoint-1"
python run.py "${COMMON_ARGS[@]}" --max_steps 1

test -f "$OUTPUT_DIR/checkpoint-1/adapter_manifest.json"
test -f "$OUTPUT_DIR/checkpoint-1/pytorch_model.bin"
python -c "import json; p='$OUTPUT_DIR/checkpoint-1/trainer_state.json'; assert json.load(open(p))['global_step'] == 1"

echo "Stage 2: reconstruct the model and resume checkpoint-1 to checkpoint-2"
python run.py "${COMMON_ARGS[@]}" --max_steps 2

test -f "$OUTPUT_DIR/checkpoint-2/adapter_manifest.json"
test -f "$OUTPUT_DIR/checkpoint-2/pytorch_model.bin"
python -c "import json; p='$OUTPUT_DIR/checkpoint-2/trainer_state.json'; assert json.load(open(p))['global_step'] == 2"

echo "PASS real Llama adapter checkpoint: $MODE checkpoint-1 -> checkpoint-2"
du -h "$OUTPUT_DIR"/checkpoint-*/pytorch_model.bin
