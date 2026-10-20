# ChatGLM3/Subj 训练数据记忆与隐私保护实验

本项目在同一个 ChatGLM3-6B 基座和固定 Subj 数据集上比较三种保护方法：Embedding、Example Steering 和 Gradient。核心问题是：生成模型是否会记忆训练原文，以及保护机制能否降低已知前缀提取攻击的成功率。

主实验只使用原始 Subj 文本或方法本身规定的保护变换。Canary 改写文本仅保留为实现诊断，不进入正式结果。

## 实验约定

| 项目 | 配置 |
|---|---|
| 基座模型 | `chatglm3-6b` |
| 模型类型 | `AutoModelForCausalLM`，ChatGLM3-6B |
| 原始数据 | `data/subj_raw_prepared/subj/none_a3c561096c597e10` |
| 数据规模 | train 7,200；validation 800；test 2,000 |
| 训练任务 | causal next-token generation |
| 主攻击 | known-prefix causal extraction |
| Holdout | 从未参与训练或 steering 构建的 test split |
| 随机种子 | 42 |

训练集和 holdout 必须分开报告。只有训练成员的提取率明显高于 holdout，才能把命中解释为训练数据记忆，而不是语言模型对常见文本的自然补全。

## 三种方法

| 方法 | 未保护目标 | 保护目标 | 保护位置 |
|---|---|---|---|
| Embedding | 原始 Subj causal LoRA | ClipText+ 文本 causal LoRA | 训练输入 |
| Example | 原文 causal LoRA + Mean Steering | 同一 LoRA + Private Steering | 推理隐藏状态 |
| Gradient | MeZO causal LoRA | DPZero causal LoRA | 零阶参数更新 |

三组实验的攻击长度和模型容量不同，因此只能在同一种方法内部比较保护前后，不能直接用成功率给三种方法排序。

### Embedding

Embedding 方法先对训练文本做 ClipText+ 嵌入保护，再使用与 raw 模型完全相同的 causal LoRA 参数训练：

```text
未保护：ChatGLM3-6B + LoRA(raw Subj)
保护后：ChatGLM3-6B + LoRA(ClipText+ Subj)
```

攻击时，两边都使用相同的原始 Subj prefix 和 reference。这样测量的是保护模型能否恢复保护前的 source-original，而不是能否恢复它实际见过的扰动文本。`attack-embedding-known-direct` 可攻击扰动文本，只用于确认模型确实学习了保护后的训练数据。

主配置为完整 7,200 条训练记录、15 epochs、LoRA rank 128、alpha 256、batch 8、learning rate `1e-4`、max length 512。

### Example Steering

Example 由三层组合构成：

```text
ChatGLM3-6B 基座 + 原始 Subj 生成式 LoRA + Mean/Private Steering
```

原始 causal LoRA 负责记忆具体句子。单个聚合 steering 向量没有独立保存 7,200 个不同续写的容量，不能替代生成式 LoRA。

对每条 Subj 记录，程序构造正确标签和错误标签两个示例，在 ChatGLM 第 11-15 层读取倒数第二个 token 的隐藏状态：

```text
d_i = hidden(correct example) - hidden(incorrect example)
```

Mean Steering 直接聚合全部 7,200 个方向：

```text
v_mean = mean(d_i) / max_record_norm
```

Private Steering 先对逐记录方向做范数裁剪，再加入高斯噪声：

```text
v_private = mean(clip(d_i, C)) / C + N(0, sigma^2)
C = 18, sigma = 0.005
```

生成时通过 forward hook 修改第 11-15 层隐藏状态：

```text
hidden' = hidden + 128 * steering_direction
```

Mean 和 Private 使用相同的基座、LoRA、攻击目标和 multiplier，只切换 steering 向量。

Private Steering 没有 epsilon accountant，不能声明为形式化 DP。它是黑盒推理输出保护；如果攻击者能绕过 hook 或直接取得底层 raw LoRA，steering 不能保护 LoRA 权重中的记忆。

### Gradient

Gradient 使用双向零阶估计训练 causal LoRA。每一步采样随机方向 `z`，分别计算正负扰动损失：

```text
g_i = (L_i(theta + mu*z) - L_i(theta - mu*z)) / (2*mu)
```

MeZO 对 batch 内投影梯度取均值并沿 `z` 更新，是未保护对照。DPZero 在取均值前对逐样本投影梯度裁剪，并加入高斯噪声：

```text
MeZO:   g = mean(g_i)
DPZero: g = mean(clip(g_i, -C, C)) + N(0, sigma^2)
```

正式 Gradient 实验仍包含全部 7,200 条原文，并从中固定选择 100 条未经修改的原始成员，每条在训练池中出现 5,000 次。训练池共有 507,100 个 occurrence。它是自然原文的高暴露压力测试，不是 Canary。

配置使用已知前 8 tokens、恢复后 4 tokens；loss 只计算后 4 tokens。LoRA 只作用于 ChatGLM 最后一层（layer 27）的 `query_key_value`，rank 1、alpha 2，共 8,704 个可训练参数。MeZO 和 DPZero 使用相同的 100 条成员、模型容量、batch 和 9,000 steps；DPZero 额外使用 `clip=0.1`、目标 `epsilon=8`、`delta=1e-5`。

由于记录被重复，DPZero accountant 的隐私单位是训练池中的单次 occurrence。该 epsilon 不能解释为重复成员的唯一记录级或用户级 DP 保证。

## 当前结果

以下结果均来自 100 个训练目标和 100 个 holdout，并通过 `compare-*` 校验两边攻击目标、prefix、reference 和生成参数一致。

| 方法 | 目标 | Exact | Fuzzy | hit@4 | hit@8 | Holdout fuzzy | Holdout hit |
|---|---|---:|---:|---:|---:|---:|---:|
| Embedding | raw | 66/100 (66%) | 67/100 (67%) | 67/100 (67%) | 67/100 (67%) | 0/100 | 0/100 |
| Embedding | ClipText+ | 0/100 | 1/100 (1%) | 1/100 (1%) | 0/100 | 0/100 | 0/100 |
| Example | Mean Steering | 15/100 (15%) | 23/100 (23%) | 42/100 (42%) | 30/100 (30%) | 0/100 | 0/100 |
| Example | Private Steering | 0/100 | 0/100 | 1/100 (1%) | 0/100 | 0/100 | 0/100 |
| Gradient | MeZO | 3/100 (3%) | 3/100 (3%) | 3/100 (3%) | N/A | 0/100 | 0/100 |
| Gradient | DPZero | 0/100 | 1/100 (1%) | 0/100 | N/A | 0/100 | 0/100 |

Gradient 只生成 4 tokens，因此不报告 hit@8。DPZero 的 1% fuzzy 来自一条 normalized-text match，但没有 exact match，也没有连续恢复 4 个 tokenizer tokens。因此严谨结论是：**Gradient exact/hit@4 从 3% 降至 0%，fuzzy 从 3% 降至 1%**，不能笼统表述为所有攻击指标都降到 0。

DPZero 完整运行 9,000 步，实际 `epsilon=7.9995`、`delta=1e-5`。正式配对结果位于：

```text
outputs/embedding_full_parallel/attacks_prefix2_random/embedding_original_comparison.json
outputs/example_reproduce/attacks/example_direct_comparison.json
outputs/gradient_natural_100_gpu23/attacks/gradient_direct_comparison.json
```

## 安装

建议使用 Python 3.11 和 CUDA 12.1：

```bash
conda create -n glm-subj python=3.11 -y
conda activate glm-subj
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHON_BIN="$(which python)"
```

以下命令均从项目根目录执行：

```bash
cd /home/wym/projects/Glm-subj
```

每个模型和攻击输出目录必须不存在或为空。重新运行时请修改输出根目录，不要覆盖已有结果。

## Embedding 复现命令

两个训练可分别放在 GPU 0 和 GPU 1 并行执行。

未保护 raw LoRA：

```bash
env MODEL_ROOT="$PWD/outputs/embedding_rerun/models" \
  LORA_TRAIN_SAMPLES=0 LORA_RANK=128 LORA_ALPHA=256 \
  LORA_BATCH_SIZE=8 EPOCHS=15 LEARNING_RATE=1e-4 MAX_LENGTH=512 \
  GPU_ID=0 bash run_privacy_audit.sh train-raw
```

保护后的 ClipText+ LoRA：

```bash
env MODEL_ROOT="$PWD/outputs/embedding_rerun/models" \
  LORA_TRAIN_SAMPLES=0 LORA_RANK=128 LORA_ALPHA=256 \
  LORA_BATCH_SIZE=8 EPOCHS=15 LEARNING_RATE=1e-4 MAX_LENGTH=512 \
  GPU_ID=1 bash run_privacy_audit.sh train-embedding
```

训练完成后，分别攻击同一批原始 Subj 文本：

```bash
env MODEL_ROOT="$PWD/outputs/embedding_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/embedding_rerun/attacks" \
  ATTACK_TARGETS=100 PREFIX_TOKENS=2 MIN_SUFFIX_TOKENS=8 MAX_NEW_TOKENS=128 \
  TEMPERATURE_START=1.4 TEMPERATURE_END=1 TOP_P=0.95 TOP_K=50 \
  GPU_ID=0 bash run_privacy_audit.sh attack-raw-direct

env MODEL_ROOT="$PWD/outputs/embedding_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/embedding_rerun/attacks" \
  ATTACK_TARGETS=100 PREFIX_TOKENS=2 MIN_SUFFIX_TOKENS=8 MAX_NEW_TOKENS=128 \
  TEMPERATURE_START=1.4 TEMPERATURE_END=1 TOP_P=0.95 TOP_K=50 \
  GPU_ID=1 bash run_privacy_audit.sh attack-embedding-original-direct

ATTACK_ROOT="$PWD/outputs/embedding_rerun/attacks" \
  bash run_privacy_audit.sh compare-embedding-direct
```

## Example 复现命令

下面直接复用已经训练成功的 raw causal LoRA。也可以把 `EXAMPLE_ADAPTER` 换成上一节新训练的 `outputs/embedding_rerun/models/raw`。

先在 GPU 0 上构建 Mean 和 Private Steering：

```bash
env MODEL_ROOT="$PWD/outputs/example_rerun/models" \
  EXAMPLE_ADAPTER="$PWD/outputs/embedding_full_parallel/models/raw" \
  EXAMPLE_SAMPLES=7200 EXAMPLE_CLIP=18 EXAMPLE_NOISE=0.005 \
  STEERING_MULTIPLIER=128 GPU_ID=0 \
  bash run_privacy_audit.sh build-example
```

构建完成后，Mean 和 Private 攻击可在 GPU 0、1 并行执行：

```bash
env MODEL_ROOT="$PWD/outputs/example_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/example_rerun/attacks" \
  EXAMPLE_ADAPTER="$PWD/outputs/embedding_full_parallel/models/raw" \
  ATTACK_TARGETS=100 PREFIX_TOKENS=2 MIN_SUFFIX_TOKENS=8 MAX_NEW_TOKENS=128 \
  TEMPERATURE_START=1.4 TEMPERATURE_END=1 TOP_P=0.95 TOP_K=50 \
  STEERING_MULTIPLIER=128 EXAMPLE_PROMPT_MODE=raw GPU_ID=0 \
  bash run_privacy_audit.sh attack-example-mean-direct

env MODEL_ROOT="$PWD/outputs/example_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/example_rerun/attacks" \
  EXAMPLE_ADAPTER="$PWD/outputs/embedding_full_parallel/models/raw" \
  ATTACK_TARGETS=100 PREFIX_TOKENS=2 MIN_SUFFIX_TOKENS=8 MAX_NEW_TOKENS=128 \
  TEMPERATURE_START=1.4 TEMPERATURE_END=1 TOP_P=0.95 TOP_K=50 \
  STEERING_MULTIPLIER=128 EXAMPLE_PROMPT_MODE=raw GPU_ID=1 \
  bash run_privacy_audit.sh attack-example-private-direct

ATTACK_ROOT="$PWD/outputs/example_rerun/attacks" \
  bash run_privacy_audit.sh compare-example-direct
```

若不复用现有 LoRA，可先执行：

```bash
env MODEL_ROOT="$PWD/outputs/example_rerun/models" \
  LORA_BATCH_SIZE=8 LORA_TRAIN_SAMPLES=0 EPOCHS=15 GPU_ID=0 \
  bash run_privacy_audit.sh train-raw
```

随后把 Example 命令中的 `EXAMPLE_ADAPTER` 改为 `$PWD/outputs/example_rerun/models/raw`。

## Gradient 复现命令

MeZO 和 DPZero 可在 GPU 2、3 并行训练。命令使用新的输出目录，且所有变量均写在单条命令中，避免换终端后丢失配置。

GPU 2，未保护 MeZO：

```bash
env BASE_MODEL=/home/wym/projects/dpfinetune/chatglm3-6b \
  MODEL_ROOT="$PWD/outputs/gradient100_rerun/models" \
  ZO_TRAIN_SAMPLES=0 ZO_EXPOSURE_SAMPLES=100 ZO_EXPOSURE_REPEATS=5000 \
  ZO_EXPOSURE_MIN_SUFFIX_TOKENS=4 ZO_LOSS_PREFIX_TOKENS=8 MAX_LENGTH=14 \
  ZO_TARGET_MODULES=query_key_value ZO_LAYER_INDICES=27 \
  ZO_LORA_RANK=1 ZO_LORA_ALPHA=2 ZO_BATCH_SIZE=64 \
  ZO_STEPS=9000 ZO_LEARNING_RATE=0.005 ZO_EPSILON=0.05 \
  ZO_EVAL_STEPS=500 ZO_EVAL_SAMPLES=100 ZO_CHECKPOINT_STEPS=500 \
  ZO_EARLY_STOPPING_PATIENCE=0 GPU_ID=2 \
  bash run_privacy_audit.sh train-gradient-mezo
```

GPU 3，保护 DPZero：

```bash
env BASE_MODEL=/home/wym/projects/dpfinetune/chatglm3-6b \
  MODEL_ROOT="$PWD/outputs/gradient100_rerun/models" \
  ZO_TRAIN_SAMPLES=0 ZO_EXPOSURE_SAMPLES=100 ZO_EXPOSURE_REPEATS=5000 \
  ZO_EXPOSURE_MIN_SUFFIX_TOKENS=4 ZO_LOSS_PREFIX_TOKENS=8 MAX_LENGTH=14 \
  ZO_TARGET_MODULES=query_key_value ZO_LAYER_INDICES=27 \
  ZO_LORA_RANK=1 ZO_LORA_ALPHA=2 ZO_BATCH_SIZE=64 \
  ZO_STEPS=9000 ZO_LEARNING_RATE=0.005 ZO_EPSILON=0.05 \
  ZO_EVAL_STEPS=500 ZO_EVAL_SAMPLES=100 ZO_CHECKPOINT_STEPS=500 \
  ZO_EARLY_STOPPING_PATIENCE=0 \
  DPZERO_CLIP=0.1 DP_EPSILON=8 DP_DELTA=1e-5 GPU_ID=3 \
  bash run_privacy_audit.sh train-gradient-dpzero
```

训练完成后攻击完整的 100 条高暴露成员和 100 条 holdout：

```bash
env MODEL_ROOT="$PWD/outputs/gradient100_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/gradient100_rerun/attacks" \
  ZO_AUDIT_DATASET=audit_exposure_data ATTACK_TARGETS=100 \
  SAMPLES_PER_PREFIX=1 PREFIX_TOKENS=8 MIN_SUFFIX_TOKENS=4 MAX_NEW_TOKENS=4 \
  TOP_K=1 TOP_P=1 TEMPERATURE_START=1 TEMPERATURE_END=1 GPU_ID=2 \
  bash run_privacy_audit.sh attack-gradient-mezo-direct

env MODEL_ROOT="$PWD/outputs/gradient100_rerun/models" \
  ATTACK_ROOT="$PWD/outputs/gradient100_rerun/attacks" \
  ZO_AUDIT_DATASET=audit_exposure_data ATTACK_TARGETS=100 \
  SAMPLES_PER_PREFIX=1 PREFIX_TOKENS=8 MIN_SUFFIX_TOKENS=4 MAX_NEW_TOKENS=4 \
  TOP_K=1 TOP_P=1 TEMPERATURE_START=1 TEMPERATURE_END=1 GPU_ID=3 \
  bash run_privacy_audit.sh attack-gradient-dpzero-direct

ATTACK_ROOT="$PWD/outputs/gradient100_rerun/attacks" \
  bash run_privacy_audit.sh compare-gradient-direct
```

训练报告必须满足以下检查，否则说明命令没有使用正式配置：

```text
train_records = 7200
training_pool_records = 507100
exposure_source_indices 数量 = 100
rank = 1
layers_to_transform = 27
max_steps = 9000
```

## 攻击指标

直接前缀攻击输出以下指标：

- `exact_success_rate`：生成 token IDs 与 reference 完全一致。
- `normalized_exact_success_rate`：文本归一化后完全一致。
- `fuzzy_success_rate`：exact/normalized exact，或连续命中至少 4 tokens 且相似度不低于 0.65。
- `hit_rate_at_N_tokens`：从 reference 开始连续命中至少 N 个 tokenizer tokens。
- `mean_token_f1`、`mean_rouge_l`、`mean_char_similarity`：宽松相似度指标。
- `train_minus_holdout`：训练成员指标减去 holdout 指标。

`target_*` 是至少一次攻击成功的不同目标数，`generation_*` 是所有采样中的命中条数。正式结果必须同时报告命中数和分母，例如 `3/100`，不能只写 `3%`。

`attack-raw`、`attack-example-mean` 等非 `-direct` 命令还会运行 causal-loss membership inference，报告 mean log-probability、Min-K% 和 zlib-normalized score 的 AUC。本文主结论使用直接前缀提取。

## 输出结构

```text
outputs/<run>/models/raw/                       # raw causal LoRA
outputs/<run>/models/embedding/                 # ClipText+ causal LoRA
outputs/<run>/models/example/mean/              # Mean Steering
outputs/<run>/models/example/private/           # Private Steering
outputs/<run>/models/gradient/mezo/             # MeZO causal LoRA
outputs/<run>/models/gradient/dpzero/            # DPZero causal LoRA
outputs/<run>/attacks/**/prefix/records.jsonl   # 每次生成的逐条结果
outputs/<run>/attacks/**/prefix/summary.json    # 聚合攻击指标
outputs/<run>/attacks/**/*comparison.json       # 配对比较
```

每个训练目录包含 `training_report.json` 和本次实际攻击 population。Gradient 高暴露成员保存在 `audit_exposure_data/`，其 `manifest.json` 记录原始 source indices 和重复次数。

## 结论边界

1. Embedding 改变了训练输入，主结果测量的是保护后模型恢复 source-original 的能力。
2. Example 保护作用于带 hook 的推理路径，不保护可被直接绕过的 raw LoRA 权重，也没有形式化 epsilon。
3. Gradient 的 100 条成员各重复 5,000 次，占训练池 occurrence 的绝大多数，是高暴露压力测试。其 epsilon 是 occurrence-level，不是唯一成员或用户级保证。
4. Canary 数据改写了 completion，只能做代码和攻击实现自检，不可替代原始 Subj 主结果。
5. 受保护方法攻击失败可能是正确结果；不能只对受保护模型放宽阈值、增加采样次数或改变攻击目标。

主入口及全部环境变量定义见 `run_privacy_audit.sh`；训练和攻击的实际参数以各输出目录中的 `training_report.json` 与 `summary.json` 为准。
