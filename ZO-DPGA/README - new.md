# ZODPGA for Llama 2 7B

This directory adapts the OPT implementation to `meta-llama/Llama-2-7b-hf`.
It covers zero-shot/ICL evaluation, regular fine-tuning, LoRA, head tuning,
linear probing, direct KV prefix tuning, MeZO, and ZODPGA.

## Setup

Request access to Llama 2 on Hugging Face, then authenticate:

```bash
huggingface-cli login
pip install -r requirements.txt
```

The first implementation remains pinned to `transformers==4.28.1`, matching the
Trainer internals vendored by the original project. Upgrading Transformers requires
first replacing the copied training loop with a small `Trainer` extension.



### zo_dpga方法 85.8% 

The exact seed-2 run that produced 85.8% dev accuracy is frozen in
`configs/lora_zo_dpga_858_reproduction.yaml`. Inspect or execute it with:

```bash
python run_lora_zo_dpga_858_reproduction.py
python run_lora_zo_dpga_858_reproduction.py --run
```

For another independent repeat, use a fresh directory instead of resuming:

```bash
CUDA_VISIBLE_DEVICES=3 python run_lora_zo_dpga_858_reproduction.py \
  --output-dir result/reproduction/lora_zo_dpga_858_seed2_repeat2 \
  --run
```



### 基准方法mezo 89.2% 

For another independent repeat, use a fresh directory instead of resuming:

```bash
CUDA_VISIBLE_DEVICES=3 python run_lora_mezo_892_reproduction.py \
    --output-dir result/reproduction/lora_mezo_892_repeat2 \
    --run
```



## 攻击

和kalman方法一致，只需要换下路径
