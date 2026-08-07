# PTCDS：SST-2 扰动数据与 LLaMA2 LoRA 微调

本目录中的正式微调流程以 `md.md` 为准。统一入口为项目根目录的 `privacy_tool.py`；它将下列命令中的模型、训练集和输出位置转换为可移植路径，不改动原始研究脚本。

## 本微调方法：基于扰动后的数据进行 LoRA 微调

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_llama2_sst2_lora_data.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --train_path ./output_PTCDS_plus_glove/SST-2/eps_6.00/sword_0.90_p_0.30/train.tsv \
  --output_dir ./outputs/llama2-sst2-lora-esp6-4.20 \
  --eval_steps 948 \
  --wandb_project llama2-sst2-lora \
  --run_name llama2-7b-sst2
```

## 对比微调方法：基于原始数据进行 base 微调

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_llama2_sst2_lora_data.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --train_path ./data/SST-2/train.tsv \
  --output_dir ./outputs/llama2-sst2-lora-base-4.20 \
  --eval_steps 948 \
  --wandb_project llama2-sst2-lora \
  --run_name llama2-7b-sst2
```

## 直接评估

直接评估与 ClipText 使用相同的两路前缀抽取流程：分别加载 PTCDS 受保护权重和未保护 base 权重，各自执行抽取并运行 `attack/member_test.py`。输出文件写入本次实验的结果目录。

## 间接评估

间接评估使用 `attack/neighbour_attack.py` 与 `attack/neighbour_attack_wo.py` 分别生成受保护/未保护模型的 train、test 得分，再由 `attack/evaluate_mia_fair_compare.py` 比较。
