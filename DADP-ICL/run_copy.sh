#!/bin/bash

# 确保脚本在出错时停止
set -e

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:.


python run_copy.py 