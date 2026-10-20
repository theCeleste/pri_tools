# 保护方法、非保护基线与直接攻击评估

本文说明当前项目正式主流程中的三组方法、对应的非保护对照，以及用于评估训练数据记忆的直接攻击方法。

项目根目录：`Glm-subj/`

## 1. 当前执行入口

正式实验由 [`run_privacy_audit.sh`](run_privacy_audit.sh) 调度，Python 命令由 [`attack_chatglm.py`](attack_chatglm.py) 分发到 `privacy_attack/` 下的模块。

`reference_implementations/` 目录不是当前正式执行路径；其中的代码仅用于方法溯源、历史实现和对照。

## 2. 保护方法与非保护基线

| 方法组 | 非保护对照 | 保护方法 | 保护位置 | 训练/构建实现 |
|---|---|---|---|---|
| Embedding | Raw Subj 文本 + 普通 causal LoRA | ClipText+ 文本 + 普通 causal LoRA | 训练输入文本 | [`privacy_attack/train_lora.py`](privacy_attack/train_lora.py) |
| Example Steering | Raw causal LoRA + Mean Steering | 同一 Raw causal LoRA + Private Steering | 推理时隐藏状态 | [`privacy_attack/build_steering.py`](privacy_attack/build_steering.py)、[`privacy_attack/steering.py`](privacy_attack/steering.py) |
| Gradient | MeZO causal LoRA | DPZero causal LoRA | 零阶参数更新 | [`privacy_attack/train_zo.py`](privacy_attack/train_zo.py) |

### 2.1 Embedding

非保护基线和保护模型都使用普通 LoRA 训练，差别只在训练输入：

- 非保护：`data/subj_raw_prepared/subj/none_a3c561096c597e10`
- 保护：`data/subj_embedding_protected`
- 训练代码：[`privacy_attack/train_lora.py`](privacy_attack/train_lora.py)
- 调度命令：

  ```bash
  bash run_privacy_audit.sh train-raw
  bash run_privacy_audit.sh train-embedding
  ```

当前主流程直接读取已经生成好的 ClipText+ 数据。ClipText+ 文本变换本身不在当前 `privacy_attack/` 主流程中重新执行；其数据、映射和参数记录在 [`data/subj_embedding_protected/manifest.json`](data/subj_embedding_protected/manifest.json) 和 `mapping.json` 中。

### 2.2 Example Steering

两种模型都使用同一个 Raw causal LoRA；区别是推理时加载的 steering 向量：

- Mean Steering：非保护对照。
- Private Steering：保护方法，逐记录方向裁剪后加入高斯噪声。

代码链路：

1. [`privacy_attack/build_steering.py`](privacy_attack/build_steering.py) 从正确/错误标签示例中提取第 11--15 层的隐藏状态差异，并生成 `mean/`、`private/` 两组向量。
2. [`privacy_attack/steering.py`](privacy_attack/steering.py) 在推理时注册 forward hook，将 steering 方向加入隐藏状态。
3. [`privacy_attack/prefix_attack.py`](privacy_attack/prefix_attack.py) 或 [`privacy_attack/causal_mia.py`](privacy_attack/causal_mia.py) 加载 steering 并执行评估。

调度命令：

```bash
bash run_privacy_audit.sh train-example
bash run_privacy_audit.sh build-example
bash run_privacy_audit.sh build-example-private
```

Private Steering 当前没有 epsilon accountant，因此不能当作形式化 DP 保证；它保护的是带 hook 的推理输出，不保护底层 Raw LoRA 权重本身。

### 2.3 Gradient

MeZO 和 DPZero 共用 [`privacy_attack/train_zo.py`](privacy_attack/train_zo.py)，通过 `--method` 选择分支：

- MeZO：对 batch 内零阶投影梯度直接取均值，不裁剪、不加 DP 噪声，是非保护基线。
- DPZero：逐样本投影梯度裁剪后取均值，并加入高斯噪声，同时使用 RDP accountant 计算 epsilon。

关键实现分支位于 `train_zo.py` 中的 `if args.method == "dpzero"`，包括噪声标定、投影梯度裁剪、噪声注入和隐私会计。

调度命令：

```bash
bash run_privacy_audit.sh train-gradient-mezo
bash run_privacy_audit.sh train-gradient-dpzero
```

正式 Gradient 实验使用原始 Subj 文本；高暴露成员通过 `audit_exposure_data/` 单独保存，用于固定攻击目标。重复 occurrence 时，DPZero 的 epsilon 是 occurrence-level 说明，不等同于唯一成员或用户级 DP 保证。

### 2.4 不属于三组主方法的辅助代码

[`privacy_attack/train_dp.py`](privacy_attack/train_dp.py) 是额外的 DPSGD baseline 入口，虽然可通过 `attack_chatglm.py train-dpsgd-baseline` 调用，但不属于 README 中三组正式保护方法的主实验矩阵，也没有被 `run_privacy_audit.sh` 的三组主流程调用。

## 3. 直接攻击评估方法

### 3.1 攻击原理

直接评估使用 **known-prefix causal extraction**：

1. 从训练成员或 holdout 中取一条完整文本。
2. 只向模型提供该文本开头的固定 token 前缀。
3. 让模型自回归生成后续 token。
4. 将生成结果与 reference 原文比较，判断模型是否恢复训练文本。

核心实现是 [`privacy_attack/prefix_attack.py`](privacy_attack/prefix_attack.py)。

其中：

- `prefix_adapter()` 评估 LoRA 模型。
- `prefix_steering()` 评估 LoRA + Mean/Private Steering。
- 输出逐条结果 `prefix/records.jsonl` 和汇总指标 `prefix/summary.json`。

### 3.2 直接攻击命令

| 被评估对象 | 直接攻击命令 |
|---|---|
| Raw 非保护 LoRA | `attack-raw-direct` |
| Embedding 模型恢复已知保护文本 | `attack-embedding-known-direct` |
| Embedding 模型恢复原始 source 文本 | `attack-embedding-original-direct` |
| Mean Steering | `attack-example-mean-direct` |
| Private Steering | `attack-example-private-direct` |
| MeZO | `attack-gradient-mezo-direct` |
| DPZero | `attack-gradient-dpzero-direct` |
| 三种保护模型一起评估 | `attack-protected-direct` |

例如：

```bash
bash run_privacy_audit.sh attack-raw-direct
bash run_privacy_audit.sh attack-embedding-original-direct
bash run_privacy_audit.sh attack-example-private-direct
bash run_privacy_audit.sh attack-gradient-dpzero-direct
```

### 3.3 各方法的 reference 定义

- Raw、MeZO、DPZero：通常对比模型实际训练过的原始文本。
- Embedding：主结果使用原始 source 文本作为 reference，测量保护模型能否恢复保护前文本；`attack-embedding-known-direct` 只用于确认模型是否学到了 ClipText+ 扰动文本。
- Example Mean/Private：使用原始 Subj 文本作为攻击 reference。
- train split 是成员集合，test split 是 holdout；两者必须分别报告，不能把 holdout 命中当作训练记忆。

### 3.4 直接攻击指标

指标由 [`privacy_attack/prefix_attack.py`](privacy_attack/prefix_attack.py) 计算：

- `exact_success_rate`：生成 token 序列与 reference 完全一致。
- `normalized_exact_success_rate`：文本归一化后完全一致。
- `fuzzy_success_rate`：达到模糊相似度阈值或连续恢复足够 token。
- `hit_rate_at_N_tokens`：从 prefix 后开始连续命中至少 N 个 tokenizer token。
- `mean_token_f1`、`mean_rouge_l`、`mean_char_similarity`：宽松相似度指标。
- `train_minus_holdout`：训练成员和 holdout 指标差异。

配对结果由 [`privacy_attack/compare.py`](privacy_attack/compare.py) 生成：

```bash
bash run_privacy_audit.sh compare-embedding-direct
bash run_privacy_audit.sh compare-example-direct
bash run_privacy_audit.sh compare-gradient-direct
```

### 3.5 Direct 与非 Direct 命令的区别

带 `-direct` 的命令只运行前缀生成攻击，核心输出位于 `prefix/`。

不带 `-direct` 的命令除了前缀生成，还会调用 [`privacy_attack/causal_mia.py`](privacy_attack/causal_mia.py) 执行 causal-loss membership inference，输出 `mia/` 目录。因此，项目主结论中的“直接评估”应以 `prefix/summary.json` 和对应的 `compare-*-direct` 结果为准。

## 4. 输出目录对应关系

```text
outputs/<run>/models/raw/                  # Raw causal LoRA，非保护
outputs/<run>/models/embedding/            # ClipText+ causal LoRA，Embedding 保护
outputs/<run>/models/example/mean/         # Mean Steering，非保护对照
outputs/<run>/models/example/private/      # Private Steering，Example 保护
outputs/<run>/models/gradient/mezo/        # MeZO，非保护对照
outputs/<run>/models/gradient/dpzero/      # DPZero，Gradient 保护
outputs/<run>/attacks/**/prefix/           # 直接前缀攻击结果
outputs/<run>/attacks/**/mia/              # 非 direct 命令产生的成员推断结果
```
