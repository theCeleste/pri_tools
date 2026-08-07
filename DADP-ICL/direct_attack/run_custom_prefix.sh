#!/bin/bash

# 确保脚本在出错时停止
set -e

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:.

# 创建输出目录（如果不存在）
mkdir -p outputs

# 运行带有自定义前缀功能的脚本
python invade_prefix_with_custom.py \
  --use-custom-prefixes \
  --custom-prefix-file example_content.txt \
  --outfile outputs/custom_prefix_samples.txt \
  --batch-size 10

# 显示运行结果
if [ -f outputs/custom_prefix_samples.txt ]; then
  echo "\n\n运行成功！生成的续写结果已保存到 outputs/custom_prefix_samples.txt"
  echo "文件大小: $(ls -lh outputs/custom_prefix_samples.txt | awk '{print $5}')"
  echo "\n前10行预览："
  head -n 10 outputs/custom_prefix_samples.txt
else
  echo "\n\n错误：未能生成输出文件 outputs/custom_prefix_samples.txt"
  exit 1
fi