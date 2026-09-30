# 无扰动评估结果输出与低 FPR 指标规划

## 1. 目标与范围

统一无扰动评估的用户可见摘要格式，并调整无扰动间接评估的固定 FPR 指标范围。

覆盖：

- 无扰动直接评估；
- 无扰动间接评估；
- 控制台摘要、历史结果查询、`result.txt` 和结构化摘要中的展示层；
- LoRA 和 ICL 两套间接评估后端及其离线重算流程。

不改变：

- 扰动评估分支；
- 直接评估的匹配算法和指标定义；
- 间接攻击分数生成、样本数量、AUC 和完整 ROC 曲线。

固定 FPR 的缩减只适用于间接评估的 TPR 指标。直接评估没有固定 FPR/TPR 指标时，不应强行添加这些字段。

## 2. 统一摘要头部

所有无扰动直接/间接评估的用户可见摘要，第一行统一包含评估类型、模型、中文微调方式和数据集：

```text
方法：直接评估 模型：chatglm3，微调方式：显式，数据集：sst2
```

间接评估示例：

```text
方法：间接评估 模型：fm9g，微调方式：隐式，数据集：wikitext2
```

展示层映射如下：

| 内部字段 | 用户可见文本 |
| --- | --- |
| `evaluation_type=direct` | `直接评估` |
| `evaluation_type=indirect` | `间接评估` |
| `finetune_type=explicit` | `显式` |
| `finetune_type=implicit` | `隐式` |

JSON 中继续保留机器可读的 `evaluation_type`、`finetune_type` 英文值，不用中文值替换内部身份字段。这样可以兼容目录、命令参数和历史查询逻辑。

## 3. 固定 FPR 指标范围

无扰动间接评估只统计以下目标 FPR：

| 目标 FPR 数值 | 展示标签 |
| ---: | --- |
| `0.01` | `1%` |
| `0.001` | `0.1%` |
| `0.0001` | `0.01%` |

不再实际计算或展示 `0.1`、`0.5`、`1.0`，即 10%、50%、100% FPR 下的 TPR。

保留 AUC、完整 ROC 曲线、成员/非成员数量，以及低 FPR 对应的实际 FPR、阈值和混淆矩阵等审计字段。由于攻击分数仍需生成，移除高 FPR 目标主要减少后处理和输出，不会显著减少 GPU 推理时间。

## 4. TPR 输出格式

TPR 统一以百分比形式输出，保留两位小数，格式固定为：

```text
TPR@1%FPR: 40.00%
TPR@0.1%FPR: 40.00%
TPR@0.01%FPR: 40.00%
```

约定：

- 使用冒号，不使用等号；
- 数值乘以 100 后输出，并带 `%`；
- 固定两位小数；
- 标签顺序固定为 `1%`、`0.1%`、`0.01%`；
- 不在同一行追加“实际 FPR”、阈值或混淆矩阵；这些字段留在 CSV/JSON 中。

示例完整摘要：

```text
方法：间接评估 模型：chatglm3，微调方式：隐式，数据集：sst2
original_text: AUC=0.829800, member=10, nonmember=500
TPR@1%FPR: 40.00%
TPR@0.1%FPR: 40.00%
TPR@0.01%FPR: 40.00%
```

## 5. 计划修改的代码位置

### 5.1 统一无扰动入口

修改 `privacy_tools/indirect_evaluation.py`：

- 将固定目标常量从六个 FPR 缩减为三个低 FPR；
- `collect_results()` 只接受三个目标行和三个 ICL TPR 字段；
- `format_summary()` 增加“方法”字段，并把 `explicit/implicit` 映射成“显式/隐式”；
- 为 TPR 添加统一百分比格式化函数，避免 LoRA 和 ICL 分别拼接出不同格式；
- 历史查询和新运行均复用同一摘要格式。

修改 `privacy_tools/direct_evaluation.py`：

- 摘要头部增加“方法：直接评估”；
- 微调方式改为中文“显式/隐式”；
- 保持现有候选数、匹配数、匹配率和匹配类型统计不变。

必要时在共享格式化辅助函数中集中维护评估类型和微调方式的映射，避免直接/间接两处产生不同文案。

### 5.2 LoRA 间接后端

修改：

- `evaluate/indirect_LoRA/run.py`：默认 `FPRS` 只保留 `0.01、0.001、0.0001`；
- `evaluate/indirect_LoRA/code/evaluate.py`：只生成三个 CSV 行，并拒绝大于 `0.01` 的固定目标；
- 后端控制台输出改为三行百分比格式，不能再输出 `target_fpr=...` 形式作为最终摘要。

CSV 仍保留 `target_fpr`、`actual_fpr`、`threshold`、`tp/fp/tn/fn` 等结构化字段。

### 5.3 ICL 间接后端

修改 `evaluate/indirect_ICL/DPICL/Indirect_attack/evaluate_attack_metrics.py`：

- 只生成 `TPR@1%`、`TPR@0.1%`、`TPR@0.01%`；
- 移除 100%、50%、10% 目标及对应打印；
- 清理当前重复的 TPR 打印逻辑；
- 保留 `metrics.json`、`roc.csv`、AUC 和样本计数。

统一入口负责把后端字段转换成最终展示格式 `TPR@<百分比>FPR: <xx.xx>%`。

## 6. 历史结果与兼容策略

- 新运行生成的 `summary.json` 只写三个低 FPR 指标。
- 读取旧的六点间接评估摘要时，只展示 `FPR <= 0.01` 的三项，不在新摘要中显示高 FPR 项。
- 一次性更新现有 `no_perturbation/indirect/*/*/*/result.txt`，保证 JSON 损坏后回退文本也使用新格式。
- 原始运行目录中的旧 CSV、JSON 和日志不删除，保留实验追溯信息。
- `summary.json` 的机器字段仍使用英文枚举；中文只用于摘要和 `result.txt` 展示。
- 旧的 `historical_results/direct/*.txt` 和 `historical_results/indirect/*.txt` 仍作为旧方法资料，不自动导入无扰动历史结果。

## 7. 文档、校验清单与测试

同步修改以下说明文件中的 FPR 范围和输出示例：

- `evaluate/indirect_LoRA/README.md`；
- `evaluate/indirect_LoRA/METHOD.md`；
- `evaluate/indirect_ICL/README.md`；
- `evaluate/indirect_ICL/METHOD.md`；
- `privacy_tools/INDIRECT_EVALUATION.md`；
- `privacy_tools/NO_PERTURBATION_HISTORY_PLAN.md` 中的展示约定。

由于后端文件会修改，更新：

- `evaluate/indirect_LoRA/checksums.json`；
- `evaluate/indirect_ICL/checksums.json`。

测试调整：

- 统一入口测试：目标集合由 6 项改为 3 项，检查直接/间接摘要头部和中文微调方式；
- LoRA 交付测试：CSV 行数改为 3，验证低 FPR 集合和百分比输出；
- ICL 交付测试：只验证三个低 FPR 指标及新格式；
- CPU 离线测试：确认离线重算和历史查询只展示三个低 FPR；
- 增加旧六点历史摘要的兼容读取测试；
- 增加输出正则检查，例如：

  ```text
  ^TPR@(1|0\.1|0\.01)%FPR: [0-9]+\.[0-9]{2}%$
  ```

## 8. 验收标准

- 直接评估摘要首行明确写出“方法：直接评估”，间接评估明确写出“方法：间接评估”；
- `explicit/implicit` 在所有用户可见摘要中分别显示为“显式/隐式”；
- 新的 LoRA CSV 恰好包含三个目标 FPR；
- 新的 ICL `metrics.json` 不包含 `TPR@100%`、`TPR@50%`、`TPR@10%`；
- 所有新摘要中的 TPR 均符合 `TPR@1%FPR: 40.00%` 形式；
- AUC、样本数和三个低 FPR 下的 TPR 数值保持正确；
- 旧历史结果仍可读取，但用户不可见摘要不再展示高 FPR 项；
- 直接/间接、显式/隐式的四类无扰动组合均通过菜单预览、历史查询和 CPU 离线测试。

