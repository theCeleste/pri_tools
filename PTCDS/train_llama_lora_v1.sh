#!/bin/bash

# LLaMA隐私保护模型 - LoRA微调脚本
# 使用LoRA解决OOM问题，大幅减少内存使用

echo "开始LLaMA隐私保护模型LoRA微调..."

# 指定使用的GPU，例如GPU 2
export CUDA_VISIBLE_DEVICES=2

# 设置模型路径和输出路径
model_save_path=./save/models/Meta-Llama-3-8B-Instruct-lora-v1
model_name_or_path=./models/Meta-Llama-3-8B-Instruct
# LoRA微调配置

CUDA_VISIBLE_DEVICES=0 python ./train_sentence_llama_camera.py \
    --model_name_or_path $model_name_or_path \
    --output_dir $model_save_path \
    --task_name sst2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --target_layer 16 \
    --eps 1.0 \
    --cluster_num 100 \
    --w_noise 0.1 \
    --w_cluster_close 0.1 \
    --w_cluster_away 0.05 \
    --data_sample_ratio 1.0 \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj v_proj k_proj o_proj gate_proj up_proj down_proj \
    --bf16 \
    --max_grad_norm 3.0 \
    --warmup_ratio 0.05 \
    --max_length 128 
    #--gradient_checkpointing \
    # --fp16 \

echo "LoRA微调完成！"

# task:0.83423|pri:-89.21312
# epoch0: 0.52
# task:0.63485|pri:-99.20076
# epoch1: 0.63
# task:0.49330|pri:-99.38926
# epoch2: 0.68