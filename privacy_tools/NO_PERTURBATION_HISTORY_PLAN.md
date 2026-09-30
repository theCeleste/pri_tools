# 无扰动直接与间接评估历史结果规划

## 1. 目标

为无扰动微调的直接评估和间接评估建立独立的历史结果目录。

每个“评估类型、模型、微调方式、数据集”组合只保留一份最新结果，同时提供结构化 JSON 和原样 TXT 两种格式：

- `summary.json`：供程序读取和展示指标。
- `result.txt`：结构化结果不可用时的最终兜底，读取时原样输出。

旧的 `historical_results/direct/*.txt`、`historical_results/indirect/*.txt` 保持不变，不自动转换为新的无扰动评估结果。

## 2. 目录结构

不使用 `<run_id>`，目录只按模型、微调方式和数据集划分：

```text
historical_results_dir/
└── no_perturbation/
    ├── direct/
    │   └── <model>/<explicit|implicit>/<dataset>/
    │       ├── summary.json
    │       └── result.txt
    └── indirect/
        └── <model>/<explicit|implicit>/<dataset>/
            ├── summary.json
            └── result.txt
```

示例：

```text
historical_results/
├── no_perturbation/direct/chatglm3/explicit/sst2/
├── no_perturbation/direct/chatglm3/implicit/sst2/
├── no_perturbation/indirect/chatglm3/explicit/sst2/
└── no_perturbation/indirect/chatglm3/implicit/sst2/
```

目录中的结果代表该组合最近一次成功评估。新结果成功后覆盖旧的 `summary.json` 和 `result.txt`；失败或中断的运行不更新历史目录。

## 3. `summary.json` 内容

摘要文件是程序读取的首选来源，至少包含以下字段：

```json
{
  "schema_version": 1,
  "baseline_type": "no_perturbation",
  "evaluation_type": "direct",
  "model": "chatglm3",
  "finetune_type": "explicit",
  "dataset": "sst2",
  "status": "completed",
  "updated_at": "2026-09-29T12:00:00+08:00",
  "backend_digest": "...",
  "results": {}
}
```

字段约定：

- `baseline_type` 固定为 `no_perturbation`，不要用 `no_noise` 代替。
- `evaluation_type` 为 `direct` 或 `indirect`。
- `model` 使用统一名称，例如 `chatglm3`、`fm9g`。
- `finetune_type` 为 `explicit` 或 `implicit`。
- `status` 只有为 `completed` 时才可作为正式历史结果读取。
- `backend_digest` 记录生成该结果时的后端版本，仅用于提示和追溯，不作为隐藏历史结果的硬过滤条件。

直接评估的 `results` 至少保存：

- 候选数
- 匹配数
- 匹配率
- 匹配类型统计

间接评估的 `results` 至少保存：

- AUC
- TPR 指标
- 成员数量
- 非成员数量

详细候选文件、PKL、CSV、日志等继续保存在 `results_dir` 的运行目录中，历史目录只承担最新结果展示和兜底读取职责。

## 4. `result.txt` 内容

`result.txt` 保存最终用户可见的文本结果，内容由评估完成时生成，格式与控制台摘要一致。

写入时直接保存最终文本；读取时不解析、不重排、不补充指标，直接输出文件内容。这样即使 `summary.json` 损坏或字段版本不兼容，也能保留人工可读结果。

建议文本只包含评估摘要，不包含临时运行目录路径，避免历史目录更新后出现失效路径。

## 5. 写入逻辑

1. 评估过程仍先写入 `results_dir` 下的独立运行目录。
2. 所有阶段成功并完成结果校验后，构造统一的 `summary` 对象和最终显示文本。
3. 根据评估类型、模型、微调方式和数据集计算历史目录：

   ```text
   historical_results_dir/no_perturbation/<direct|indirect>/<model>/<explicit|implicit>/<dataset>/
   ```

4. 创建临时 JSON 和 TXT 文件，写入完成后再替换正式文件，避免留下半写入内容。
5. 先更新 `result.txt`，再更新 `summary.json`；读取方以有效的 `summary.json` 为主，必要时回退到 TXT。
6. 不创建新的 `<run_id>` 目录，不覆盖 `results_dir` 中的原始运行记录。

## 6. 读取逻辑

历史查询根据当前选择直接定位唯一目录，不扫描多个运行目录，也不按时间排序：

```text
historical_results_dir/no_perturbation/<direct|indirect>/<model>/<explicit|implicit>/<dataset>/
```

读取顺序如下：

1. 读取 `summary.json`。
2. 检查 JSON 可解析，且 `status` 为 `completed`。
3. 核对 `baseline_type`、`evaluation_type`、`model`、`finetune_type` 和 `dataset`。
4. 字段完整时，按结构化指标展示结果。
5. 如果 `summary.json` 缺失、损坏、字段不完整或身份不匹配，则读取 `result.txt`。
6. `result.txt` 存在时直接原样打印，不尝试从文本中提取指标。
7. 两个文件都不存在时，提示当前组合没有历史结果。

历史查询不再读取以下旧版文件：

```text
historical_results_dir/direct/*.txt
historical_results_dir/indirect/*.txt
```

这些文件继续作为旧方法历史资料保留，但不参与新的无扰动直接/间接评估查询。

## 7. 版本兼容原则

- 后端源码更新后，旧的无扰动历史结果仍然可展示；`backend_digest` 只作为来源信息或版本提示。
- 只有在执行离线重算时，才严格校验原始文件、结果身份和后端版本。
- `summary.json` 增加字段时保持向后兼容；无法兼容的摘要自动回退到 `result.txt`。
- 新目录中的结果不能从旧的按方法 `.txt` 文件推断模型、微调方式或数据集。

## 8. 实现范围

需要调整的主要位置：

- `privacy_tools/direct_evaluation.py`
  - 增加无扰动历史目录定位。
  - 将成功摘要和最终文本写入对应目录。
  - 修改历史读取逻辑，支持 JSON 优先、TXT 回退。
- `privacy_tools/indirect_evaluation.py`
  - 增加同样的无扰动历史目录定位、写入和回退读取逻辑。
- `privacy_tools/unified.py`
  - 菜单中的“查看最近历史结果”继续调用统一入口，不再依赖旧方法 ID。
- `privacy_tools/settings.example.json`
  - 保持 `historical_results_dir` 为总目录配置，不新增模型或运行编号配置。

实现完成后，应覆盖以下四类组合：

| 评估类型 | 显式微调 | 隐式微调 |
| --- | --- | --- |
| 直接评估 | 支持 | 支持 |
| 间接评估 | 支持 | 支持 |

