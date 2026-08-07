#!/bin/bash

# 确保脚本在出错时停止
set -e

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:.


python run_exp_sst2_loca_monitor.py \
    --sigma 0.41 \
    --openai-model "meta-llama/Llama-2-7b-hf" \
    --num-private-train 20 \
    --set-num-public-train 0 \
    --num-valid 10 \
    --num-private-train-splits 10 \
    --num-test 20 \
    --use-dp-prompts \
    --sample-same-label-prompts \
    --subsample-per-token \
    --synth-seed 0 \
    --eval-seed 0 