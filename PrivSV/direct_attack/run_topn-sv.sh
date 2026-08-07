#!/bin/bash

# 确保脚本在出错时停止
set -e

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:.

# 创建输出目录（如果不存在）
mkdir -p outputs

python extract_topn.py \
 --N 5000 \
 --batch_size 20 \
 --outfile outputs/top_n_samples-sv.txt