# SST-2 + PTCDS+ + LLaMA2 LoRA

本项目用于在 SST-2 数据集上，先使用 PTCDS+ 进行文本扰动，再基于扰动数据和原始数据分别进行 LLaMA2 LoRA 微调。

## 运行步骤

### 1. 使用 PTCDS+ 生成扰动数据

```bash
python run_PTCDS.py \
  --task SST-2 \
  --method PTCDS_plus \
  --epsilon 6.0 \
  --word_embedding_path ./data/glove.840B.300d.txt \
  --word_embedding_size 300 \
  --data_dir ./data/SST-2/ \
  --output_dir ./output_PTCDS_plus_glove/SST-2/ \
  --threads 12 \
  --p 0.3 \
  --sensitive_word_percentage 0.9
```

### 2. 基于扰动后的数据进行 LoRA 微调

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_llama2_sst2_lora_data.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --train_path ./output_PTCDS_plus_glove/SST-2/eps_6.00/sword_0.90_p_0.30/train.tsv \
  --output_dir ./outputs/llama2-sst2-lora-esp6-4.20 \
  --eval_steps 948 \
  --wandb_project llama2-sst2-lora \
  --run_name llama2-7b-sst2
```

### 3. 基于原始数据进行 base 微调

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_llama2_sst2_lora_data.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --train_path ./data/SST-2/train.tsv \
  --output_dir ./outputs/llama2-sst2-lora-base-4.20 \
  --eval_steps 948 \
  --wandb_project llama2-sst2-lora \
  --run_name llama2-7b-sst2
```
