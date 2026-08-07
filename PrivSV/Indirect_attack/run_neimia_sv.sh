# 确保脚本在出错时停止
set -e

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:.

# python neighbour_mia_sv.py \
#     --proc-id 0 \
#     --model bert \
#     --dataset sst2

python neighbour_mia_sv.py \
    --proc-id 0 \
    --model bert \
    --dataset other