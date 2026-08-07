# 一、微调
## 准备预训练词向量文件
确保/home/wrc/disk_optimizer/ClipText/sst/ClipText/ClipText/sst2路径下有glove_840B-300d.txt文件
## 准备环境

```python
# create a virtual environment (e.g. conda)
conda create -n clip-dp python=3.11
conda activate clip-dp

# install the requirements
pip install -r requirements.txt


export HF_ENDPOINT=https://hf-mirror.com
```

## 开始
## 不启用停用词的clip方法
sh clip.sh

## 启用停用词的clip+方法
sh clip+.sh

- 得到/home/wrc/disk_optimizer/ClipText/sst/ClipText/ClipText/sst2/privatized_dataset/glove_840B-300d/conservative/eps_1.0_top_20_s1_save_stop_words_False目录下脱敏后的train,test与dev文件


# 二、微调
## 准备
cd到ClipText/dp目录下

python cliptext.py \
    --gpu_id 1 \
    --model_name "/path/to/your/llama-2-7b-hf" \ #选择llama-2-7b-hf路径
    --train_data_path "/path/to/your/train_dataset" \ #选择脱敏后的训练集路径
    --val_data_path "/path/to/your/val_dataset.tsv" \ #选择验证集路径
    --batch_size 32 \
    --seq_length 128 \
    --learning_rate 1e-4 \
    --epochs 3 \
    --lora_rank 16

得到ClipText/dp/lora_outputs路径下的模型

# 三、攻击
## 准备

cd到你的模型路径下
cp -r /home/ljd/home/xx3/TextObfuscator-main/models 到ClipText/clip-attack路径下(间接攻击的掩码补全模型)

## 1.直接攻击

- 有隐私保护
修改代码200行与202行，加载经过隐私保护微调的被攻击模型
    MODEL_GPT2 = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kaldp/lora_kaldp_sst2_model_best", config)#修改为自己的模型路径
    MODEL_GPT2_XL = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kaldp/lora_kaldp_sst2_model_best", config)#修改为自己的模型路径
之后
python extraction_temperature_decay.py

- 无隐私保护
修改代码198行与200行，加载未经过隐私保护微调的被攻击模型
    MODEL_GPT2 = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kaldp/lora_wo_sst2_model_best", config)#修改为自己的模型路径
    MODEL_GPT2_XL = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kaldp/lora_wo_sst2_model_best", config)#修改为自己的模型路径
之后
python extraction_temperature_decay_wo.py

将得到的文件检测
修改代码6行与24行，加载训练集和得到的文件
    tsv_path = "/home/wrc/full_model/datasets/SST-2/train.tsv" #修改为使用的训练集路径
    attack_path = "/home/wrc/kaldp_model/save/results/extraction_temperature_decay.txt"#修改为得到的文件的路径
python member_test.py
发现有隐私保护攻击成功率为0，无隐私保护攻击出了一条记录 => 降低100%

## 2.间接攻击
- 有隐私保护
修改代码186行，加载经过隐私保护微调的被攻击模型
    attack_model = load_model_for_causal_lm("/home/wrc/meta-llama/Llama-2-7b-hf", device,"/home/wrc/disk_optimizer/kaldp/lora_kaldp_sst2_model_best", config)#修改为自己的模型路径
修改代码67,73,80行，加载数据集
        data_files="/home/wrc/full_model/datasets/SST-2/train.tsv", #修改为目标训练集路径
        data_files="/home/wrc/full_model/datasets/SST-2/test.tsv", #修改为目标测试集路径
        data_files="/home/wrc/full_model/datasets/SST-2/dev.tsv", #修改为目标验证集路径

之后
python neighbour_attack.py --proc-id 0 --model roberta --dataset sst2  --split train
生成all_scores_sst2_roberta_0_train.pkl
python neighbour_attack.py  --proc-id 0 --model roberta --dataset sst2  --split test
生成all_scores_sst2_roberta_0_test.pkl

- 无隐私保护
修改代码190行，加载未经过隐私保护微调的被攻击模型
    attack_model = load_model_for_causal_lm("/home/wpj/model/llama2-7b", device, "/home/wrc/disk_optimizer/kaldp/lora_wo_sst2_model_best", config) #修改为自己的模型路径
    修改代码69,75,82行，加载数据集
        data_files="/home/wrc/full_model/datasets/SST-2/train.tsv", #修改为目标训练集路径
        data_files="/home/wrc/full_model/datasets/SST-2/test.tsv", #修改为目标测试集路径
        data_files="/home/wrc/full_model/datasets/SST-2/dev.tsv", #修改为目标验证集路径
之后
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split train
生成wo_all_scores_sst2_roberta_0_train.pkl
python neighbour_attack_wo.py --proc-id 0 --model roberta --dataset sst2 --split test
生成wo_all_scores_sst2_roberta_0_test.pkl

- 得到对比结果（分析间接攻击生成的pkl文件）
python evaluate_mia_fair_compare.py   --base-train-pkl wo_all_scores_sst2_roberta_0_train.pkl   --base-test-pkl  wo_all_scores_sst2_roberta_0_test.pkl   --priv-train-pkl all_scores_sst2_roberta_0_train.pkl   --priv-test-pkl  all_scores_sst2_roberta_0_test.pkl   --target-fprs 0.025 0.01   --pos-neg-ratio 0.2