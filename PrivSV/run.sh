# 确保脚本在出错时停止
set -e

# 设置环境变量
# export HF_ENDPOINT=https://hf-mirror.com
# export PYTHONPATH=$PYTHONPATH:.
# sleep 7h

python run.py \
--model "meta-llama/Llama-2-7B-chat-hf" \
--dataset "sst2" \
--layers 11 12 13 14 15 \
--noise_multiplier 0.06 \
--clip 18 