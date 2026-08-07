# 基于噪声自适应过滤的差分隐私梯度扰动微调方法文档

# <保护方法>

## 1. 准备环境

```python
# create a virtual environment (e.g. conda)
conda create -n kal-dp python=3.11
conda activate kal-dp

# install the requirements
pip install -r requirements.txt


export HF_ENDPOINT=https://hf-mirror.com
```

## 2. Kal-Dp隐私保护微调方法
```
CUDA_VISIBLE_DEVICES=0 python kal_dp.py \\
    --batch_size 32 \
    --max_physical_batch_size 16 \
    --seq_length 128 \
    --is_lora True \
    --lora_rank 16 \
    --learning_rate 0.0001 \
    --max_grad_norm 3.0 \
    --epochs 2 \
    --model_name "/home/wrc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" \
```
- 得到kal_dp/lora_kal_dp_sst2_model_best目录下的微调模型

## 3. 标准隐私保护微调方法DPSGD
```
CUDA_VISIBLE_DEVICES=2 python dpsgd.py \
    --batch_size 32 \
    --max_physical_batch_size 12 \
    --seq_length 128 \
    --is_lora True \
    --lora_rank 16 \
    --learning_rate 0.0001 \
    --max_grad_norm 3.0 \
    --epochs 2 \
    --model_name "/home/wrc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" \
```
- 得到kal_dp/lora_dp_sst2_model_best目录下的微调模型

## 4. 无隐私保护微调方法
```
python Peft.py \
    --batch_size 32 \
    --seq_length 128 \
    --train_data_path "/home/wrc/sstdp/CusText/CusText/sst2/datasets/sst2/train.tsv" \
    --val_data_path "/home/wrc/sstdp/CusText/CusText/sst2/datasets/sst2/dev.tsv" \
    --lora_rank 16 \
    --learning_rate 0.0001 \
    --epochs 3 \
    --eval_every_n_steps 400 \
    --gpu_id 2 \
    --model_name "/home/wrc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" \
```
- 得到kal_dp/lora_outputs目录下对应目录的微调模型

# <攻击方法>
## 准备

cd到你的模型路径下
cp -r /home/ljd/home/xx3/TextObfuscator-main/models 到kal_dp/kal_dp_attack路径下(间接攻击的掩码补全模型)

## 1.直接攻击

- 有隐私保护
修改代码200行与202行，加载经过隐私保护微调的被攻击模型
    MODEL_GPT2 = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kal_dp/lora_kal_dp_sst2_model_best", config)#修改为自己的模型路径
    MODEL_GPT2_XL = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kal_dp/lora_kal_dp_sst2_model_best", config)#修改为自己的模型路径
之后
python extraction_temperature_decay.py

- 无隐私保护
修改代码198行与200行，加载未经过隐私保护微调的被攻击模型
    MODEL_GPT2 = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kal_dp/lora_wo_sst2_model_best", config)#修改为自己的模型路径
    MODEL_GPT2_XL = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kal_dp/lora_wo_sst2_model_best", config)#修改为自己的模型路径
之后
python extraction_temperature_decay_wo.py

将得到的文件检测
python member_test.py
发现有隐私保护攻击成功率为0，无隐私保护攻击出了一条记录 => 降低100%

## 2.间接攻击
- 有隐私保护
修改代码186行，加载经过隐私保护微调的被攻击模型
    attack_model = load_model_for_causal_lm("/home/wrc/meta-llama/Llama-2-7b-hf", device,"/home/wrc/disk_optimizer/kal_dp/lora_kal_dp_sst2_model_best", config)#修改为自己的模型路径
之后
python neighbour_attack.py --proc-id 0 --model roberta --dataset sst2  --split train
生成all_scores_sst2_roberta_0_train.pkl
python neighbour_attack.py  --proc-id 0 --model roberta --dataset sst2  --split test
生成all_scores_sst2_roberta_0_test.pkl

- 无隐私保护
修改代码190行，加载未经过隐私保护微调的被攻击模型
    attack_model = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kal_dp/lora_wo_sst2_model_best", config) #修改为自己的模型路径
之后
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split train
生成wo_all_scores_sst2_roberta_0_train.pkl
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split test
生成wo_all_scores_sst2_roberta_0_test.pkl

- 得到对比结果（分析间接攻击生成的pkl文件）
python evaluate_mia_fair_compare.py   --base-train-pkl wo_all_scores_sst2_roberta_0_train.pkl   --base-test-pkl  wo_all_scores_sst2_roberta_0_test.pkl   --priv-train-pkl all_scores_sst2_roberta_0_train.pkl   --priv-test-pkl  all_scores_sst2_roberta_0_test.pkl   --target-fprs 0.025 0.01   --pos-neg-ratio 0.2