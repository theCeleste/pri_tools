# lora微调

## 统一直接评估入口

ChatGLM3-6B 与 FM9G-7B 的 SST-2/WikiText-2 直接评估由
`privacy_tools.direct_evaluation` 统一编排。预览不会加载模型或创建结果目录：

```bash
python -m privacy_tools.direct_evaluation \
  --model chatglm3 --finetune explicit --dataset sst2 --mode preview
```

正式运行将候选、匹配结果、配置、命令、环境和 `summary.json` 保存到
`privacy_tools/results/direct_evaluation/<model>/<finetune>/<dataset>/<run_id>/`。
成功运行还会把最新摘要和文本兜底写入
`privacy_tools/historical_results/no_perturbation/direct/<model>/<finetune>/<dataset>/`，其中包含
`summary.json` 和 `result.txt`，不使用 `<run_id>`。
历史查询使用 `--mode history`，优先读取历史目录的 `summary.json`，文件缺失或无效时原样输出 `result.txt`；对已有运行目录离线重算使用
`--mode evaluate --source-run <run_dir>`。

直接评估使用独立解释器配置：显式 LoRA 读取 `python.direct_lora`，隐式 ICL
读取 `python.direct_icl`；两者可分别指向与间接评估对应的不同环境。
`evaluation.direct.datasets.wikitext2.path` 支持本地 DatasetDict、
`train/validation/test.json`（或 JSONL）、文本文件以及 parquet 分片；成员匹配只读取训练划分。

## chatglm

### 1. 生成候选样本（temperature_chatglm.py）

按temperature 采样，生成1000条候选样本，计算每条样本的 PPL 和 Zlib 压缩率，按 PPL/Zlib 对数比值排序，取前100条

生成1000条样本与前100条样本

```
python temperature_chatglm.py \
    --base_model_path ../chatglm3-6b \   #修改为自己的chatglm路径
    --lora_path .../chatglm3_sst2_memorize_final \ #修改lora路径
    --dataset sst2 \
    --N 1000 \
    --batch_size 5 \
    --seq_len 180 \
    --top_p 0.6 \
    --top_k 20 \
    --repetition_penalty 1.0 \
    --dtype auto \
    --min_chars 20 \
    --outfile ./save/results/chatglm_sst2 \
    --verbose
```

输出

- `chatglm_sst2_all.txt` / `_all.json`：全部 1000 条
- `chatglm_sst2_top100.txt`：前 100 条（按 ratio 升序）

`--dataset` 可选 `sst2` / `wikitext`，需搭配对应 LoRA。

### 2. 记忆检测（sst2_member_test.py）

对于生成的100条样本进行记忆检测，对每条生成样本，依次尝试 4 种匹配：

- raw exact（原文完全一致）
- normalized exact （归一化后完全一致）
- contains （互为子串）（长度 ≥ 50）

fuzzy (difflib)  ：相似度 ≥ 0.85

统计匹配率、匹配类型分布、类别分布，输出 JSON / CSV / TXT / summary / unmatched

```
需将代码处换成自己的路径：
train_dataset_path = ".../finetune/data/sst2/train"
attack_path = ".../save/results/llama_sst2/top100_by_ppl_low_to_high_20260714_134512.txt"
save_dir = ".../save/results/llama_sst2/ppl"
```

```
python sst2_member_test.py
```

## 9g

### 1.生成候选样本（temperature_9g.py）

```
python 你的脚本文件名.py \
    --dataset agnews \
    --base_model_path .../9G7B_MHA \
    --lora_path ../checkpoints \
    --N 1000 \
    --batch_size 10 \
    --seq_len 180 \
    --top_p 0.6 \
    --top_k 20 \
    --decode_mode sample \
    --dtype auto \
    --outfile ./save/results/9g_sst2
```

输出

- `chatglm_sst2_all.txt` / `_all.json`：全部 1000 条
- `chatglm_sst2_top100.txt`：前 100 条（按 ratio 升序）

`--dataset` 可选 `sst2` / `wikitext`，需搭配对应 LoRA。

### 2.记忆检测

与chatglm相同



# 示例微调

### 1.生成示例文件(json) -run_exp_sst2_local.py

- chatglm-sst2

```
python run_exp_sst2_local.py \
  --task sst2 \
  --data-dir ../data/sst2 \
  --openai-model ./chatglm3-6b \
  --device cuda:1 \
  --num-valid 100 \
  --num-test 100 \
  --max-token-per-text 64 \
  --eval-seed 0 \
  --output-file ./results/chatglm-sst2/synthetic_text_results.json
```

- chatglm-wikitext

```
python run_exp_sst2_local.py \
  --task wikitext \
  --data-dir ../data/wikitext_2_raw_v1 \
  --openai-model ./chatglm3-6b \
  --device cuda:1 \
  --wikitext-query-split train \
  --wikitext-min-words 8 \
  --num-valid 100 \
  --num-test 100 \
  --max-token-per-text 64 \
  --eval-seed 0 \
  --output-file ./results/chatglm-wikitext/synthetic_text_results.json
```

- 9g-sst2

```
python run_exp_sst2_local.py \
  --task sst2 \
  --data-dir ../data/sst2 \
  --openai-model ./9G7B_MHA \
  --device cuda:1 \
  --num-valid 100 \
  --num-test 100 \
  --max-token-per-text 64 \
  --eval-seed 0 \
  --output-file ./results/9g-sst2/synthetic_text_results.json
```

- 9g-wikitext

```
python run_exp_sst2_local.py \
  --task wikitext \
  --data-dir ../data/wikitext_2_raw_v1 \
  --openai-model ./9G7B_MHA \
  --device cuda:1 \
  --wikitext-query-split train \
  --wikitext-min-words 8 \
  --num-valid 100 \
  --num-test 100 \
  --max-token-per-text 64 \
  --eval-seed 0 \
  --output-file ./results/9g-wikitext/synthetic_text_results.json
```

### 2.记忆检测-member_test_current.py

- sst2

```
python member_test_current.py \
  --task sst2 \   
  --generated-json ./results/chatglm-sst2/synthetic_text_results.json 
  --train-data ../data/sst2 \ 
  --candidate-field synthetic_text \
  --output-dir ./results/chatglm-sst2/member_test
```

- wikitext

```
python direct_attack/member_test_current.py \
  --task wikitext \
  --generated-json ./results/chatglm-wikitext/synthetic_text_results.json \
  --train-data ../data/wikitext_2_raw_v1 \
  --candidate-field generated_text \
  --window-min-tokens 8 \
  --wikitext-max-candidates 300 \
  --fuzzy-threshold 0.85 \
  --output-dir ./results/chatglm-wikitext/member_test
```

