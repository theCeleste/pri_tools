# 统一入口 UI 与任务架构改造规划

本文档只描述规划，不在当前阶段修改六种方法的训练代码、攻击代码或实际运行逻辑。

## 1. 改造目标

将当前“先选方法类别，再选方法，再选四类运行项”的入口，改造成面向用户任务的两级入口：

```text
1. 微调
2. 评估
```

用户只需要选择任务相关内容；模型、数据集和实验参数默认继续从 JSON 配置读取。底层训练脚本、攻击脚本和已有命令构造逻辑尽量复用。

入口文件 `privacy_tool.py` 保持不变，仍然调用 `privacy_tools.unified.main()`。

## 2. 总体流程

```text
privacy_tool.py
    ↓
privacy_tools.unified.main()
    ↓
选择任务：微调 / 评估
    ├─ 微调流程
    └─ 评估流程
    ↓
生成统一的任务选择结果
    ↓
读取 JSON 配置并应用用户选择的覆盖值
    ↓
环境、模型、数据集和资源检查
    ↓
确认
    ↓
复用现有 catalog builder 生成 Step
    ↓
core.run_experiment()
```

## 3. 微调流程

### 3.1 菜单层级

```text
微调
 ├─ 梯度扰动
 │   ├─ Kal-DP
 │   └─ ZO-DPGA
 ├─ 嵌入扰动
 │   ├─ ClipText
 │   └─ PTCDS
 └─ 示例扰动
     ├─ DADP-ICL
     └─ PrivSV
       ↓
选择模型
选择数据集
       ↓
选择当前方法 / 对比方法
       ↓
执行训练
```

### 3.2 方法映射

| 扰动类别 | 具体方法 | 当前方法实验 ID | 对比方法实验 ID |
|---|---|---|---|
| 梯度扰动 | Kal-DP | `utility` | `baseline` |
| 梯度扰动 | ZO-DPGA | `utility` | `baseline` |
| 嵌入扰动 | ClipText | `utility` | `baseline` |
| 嵌入扰动 | PTCDS | `utility` | `baseline` |
| 示例扰动 | DADP-ICL | `utility` | `baseline` |
| 示例扰动 | PrivSV | `utility` | `baseline` |

这里的“当前方法/对比方法”不是新的算法类型，而是对现有 `utility` 和 `baseline` 实验的用户级封装。

### 3.3 模型和数据集

模型和数据集选择位于方法选择之后、当前/对比方法选择之前。

默认值保持当前设置：

```text
模型：meta-llama/Llama-2-7b-hf
数据集：SST-2
```

当前实现只开放 JSON 中已经配置的选项：

```text
模型：
  - 当前 JSON 中的 Llama-2-7B 默认模型

数据集：
  - 当前 JSON 中的数据集（默认 SST-2）
```

模型和数据集菜单只用于确认 JSON 配置值，自定义输入选项暂时关闭。具体路径仍由配置和方法 builder 解析。

模型和数据集的优先级应统一为：

```text
命令行 --set 覆盖
    > JSON 配置
    > 内置默认值
```

其他实验参数和各方法 Python 解释器也直接读取 JSON；UI 不再提供逐项参数编辑或环境切换。JSON 指定的环境缺少依赖时停止执行并提示用户修改配置。

### 3.4 微调结果

执行完成后继续复用当前训练结果机制：

- 原始 stdout 写入 `stdout.log`
- 运行清单写入 `run_manifest.json`
- 训练指标追加到 `experiment_metrics.csv`
- 展示准确率、峰值内存、运行时间、吞吐率等指标
- 读取并展示当前方法对应的历史性能结果

真实执行时：

- 选择“当前方法”时，在历史性能结果末尾提示：

  ```text
  其中基准方法结果为历史结果
  ```

- 选择“对比方法”时，在历史性能结果末尾提示：

  ```text
  其中本方法结果为历史结果
  ```

安全预演不显示上述提示。

## 4. 评估流程

### 4.1 第一层：评估类型

```text
评估
 ├─ 直接评估
 └─ 间接评估
```

对应现有实验 ID：

| 用户选择 | 现有实验 ID |
|---|---|
| 直接评估 | `direct_attack` |
| 间接评估 | `indirect_attack` |

### 4.2 第二层：评估模型的微调类型

```text
直接评估 / 间接评估
       ↓
选择评估模型的微调类型
       ├─ 显式微调模型
       └─ 隐式微调模型
```

“显式微调模型”和“隐式微调模型”是两个不同分支。

### 4.3 显式微调模型

```text
显式微调模型
 ├─ 梯度扰动
 │   ├─ Kal-DP
 │   └─ ZO-DPGA
 ├─ 嵌入扰动
 │   ├─ ClipText
 │   └─ PTCDS
 └─ 无扰动
```

映射如下：

| 用户选择 | 当前实现 |
|---|---|
| 梯度扰动/Kal-DP | `kal_dp` 的评估 builder |
| 梯度扰动/ZO-DPGA | `zo_dpga` 的评估 builder |
| 嵌入扰动/ClipText | `cliptext` 的评估 builder |
| 嵌入扰动/PTCDS | `ptcds` 的评估 builder |
| 无扰动 | 仅保留接口，暂不执行 |

### 4.4 隐式微调模型

```text
隐式微调模型
 ├─ 示例扰动
 │   ├─ DADP-ICL
 │   └─ PrivSV
 └─ 无扰动
```

映射如下：

| 用户选择 | 当前实现 |
|---|---|
| 示例扰动/DADP-ICL | `dadp_icl` 的评估 builder |
| 示例扰动/PrivSV | `privsv` 的评估 builder |
| 无扰动 | 仅保留接口，暂不执行 |

### 4.5 无扰动接口

无扰动选项必须出现在菜单中，但当前不启动训练或评估命令。

选择后输出：

```text
无扰动微调评估功能暂不在本系统需求范围内，当前不执行。
```

随后安全返回菜单或结束本次任务，不创建正式实验目录，不写入实验指标。

实现上建议返回一个明确的 `unsupported`/`not_implemented` 状态，而不是伪造一个 `Method` 或空的 `Step` 列表，避免被误认为实验成功。

### 4.6 评估模型和参数

评估阶段最后选择模型，默认仍使用：

```text
meta-llama/Llama-2-7b-hf
```

其他参数继续从 JSON 读取，包括：

- adapter 路径
- baseline adapter 路径
- steering checkpoint
- mask model
- 攻击样本数
- 邻居数量
- GPU
- Python 解释器

直接评估和间接评估的历史结果读取规则：

```text
直接评估：historical_results/direct/<方法>.txt
间接评估：historical_results/indirect/<方法>.txt
```

评估阶段只展示当前选择的方法对应文件，不展示其他方法的历史结果。

## 5. 推荐的内部选择对象

不建议让 `interactive()` 只返回一个方法 ID。建议生成一个统一的选择对象，例如：

```python
TaskSelection(
    task_kind="finetune" | "evaluation",
    evaluation_kind="direct_attack" | "indirect_attack" | None,
    finetune_type="explicit" | "implicit" | None,
    perturbation_group="gradient" | "embedding" | "example" | "none",
    method_id="kal_dp" | "zo_dpga" | "cliptext" | "ptcds" | "dadp_icl" | "privsv" | "none",
    experiment_id="utility" | "baseline" | "direct_attack" | "indirect_attack" | None,
    model_override=str | None,
    dataset_override=str | None,
)
```

其中：

- 微调任务的 `experiment_id` 根据“当前方法/对比方法”分别为 `utility`/`baseline`。
- 评估任务的 `experiment_id` 根据直接/间接评估分别为 `direct_attack`/`indirect_attack`。
- 无扰动任务的 `method_id` 可以为 `none`，并在执行前被识别为暂不支持。

这样可以把 UI 选择与底层 `Method`/`Experiment` 解耦，避免继续在 `interactive()` 中堆叠大量条件分支。

## 6. 计划修改的文件

### 必改

#### `privacy_tools/unified.py`

- 重写 `interactive()` 的菜单顺序。
- 增加“微调/评估”顶层选择。
- 增加显式/隐式微调模型选择。
- 增加模型和数据集选择。
- 增加微调时的“当前方法/对比方法”选择。
- 保留现有 JSON 参数加载逻辑。
- 将新的 `TaskSelection` 映射为现有 `Method` 和 `Experiment`。
- 对无扰动分支做提前返回。

#### `privacy_tools/catalog.py`

- 抽取并集中维护扰动类别分组。
- 增加显式/隐式评估支持关系。
- 为每个方法标记支持的任务类型。
- 保留现有训练和评估 builder，优先不改底层命令。

### 视实现需要修改

#### `privacy_tools/core.py`

- 如有必要增加 `TaskSelection` 数据结构。
- 展示并使用 JSON 中的模型和数据集。
- 保持 `run_experiment()` 接口兼容。
- 保持历史性能/直接评估/间接评估结果展示逻辑。
- 增加“未实现任务”的明确返回状态。

#### `privacy_tools/settings.example.json`

如果模型和数据集要作为统一 UI 选项管理，建议增加：

```json
{
  "common": {
    "dataset": "sst2",
    "model_7b": "meta-llama/Llama-2-7b-hf"
  }
}
```

已有字段能够满足需求时，不必重复新增配置。

#### `privacy_tools/bootstrap.py`

仅在模型/数据集选择需要新增检查逻辑时修改。例如：

- 根据选择的数据集检查数据文件。
- 根据选择的模型检查本地路径或 Hugging Face 标识。

### 必须同步检查但不一定改动

- `smoke_test.py`
- `full_audit.py`
- `README.md`
- `privacy_tools/README.md`

这些文件当前假设每种方法固定四个运行项、总计 24 项。改造后需要把审计重点从“固定菜单结构”调整为“所有任务分支均能正确映射”。

## 7. 命令行兼容策略

现有命令行接口建议暂时保留：

```bash
python3 privacy_tool.py list
python3 privacy_tool.py doctor --method kal_dp
python3 privacy_tool.py run --method kal_dp --experiment utility --dry-run
```

交互 UI 使用新流程，命令行 `run` 继续作为底层调试和自动化入口。这样不会破坏已有脚本和 smoke test。

后续如果需要命令行也采用新语义，可以另行增加：

```bash
python3 privacy_tool.py run --task finetune ...
python3 privacy_tool.py evaluate --type direct ...
```

不建议在第一阶段直接删除旧参数。

## 8. 历史结果和输出策略

### 微调

继续使用：

```text
privacy_tools/results/<类别>/<方法>/<utility|baseline>/...
```

历史性能结果：

```text
privacy_tools/historical_results/<方法>.txt
```

### 评估

继续使用当前运行目录保存实际评估产物：

```text
privacy_tools/results/<类别>/<方法>/<direct_attack|indirect_attack>/...
```

历史评估结果：

```text
privacy_tools/historical_results/direct/<方法>.txt
privacy_tools/historical_results/indirect/<方法>.txt
```

评估历史结果只用于终端展示，不覆盖当前运行生成的评估文件。

## 9. 分阶段实施顺序

### 阶段一：抽取元数据

1. 在 `catalog.py` 中整理三类扰动方法。
2. 整理显式/隐式评估映射。
3. 标记无扰动为接口项。
4. 不改变现有 builder。

### 阶段二：重写交互选择

1. 增加顶层“微调/评估”。
2. 实现微调路径。
3. 实现评估路径。
4. 增加模型、数据集选择。
5. 增加当前方法/对比方法选择。
6. 将所有未选择参数交给 JSON 配置。

### 阶段三：统一选择结果到实验映射

1. 引入 `TaskSelection`。
2. 将 `utility`、`baseline`、`direct_attack`、`indirect_attack` 作为兼容层。
3. 增加无扰动提前返回。
4. 验证历史结果展示使用正确的根目录或子目录。

### 阶段四：更新检查和文档

1. 更新 smoke test。
2. 更新 full audit 的结构假设。
3. 更新 README 菜单说明。
4. 增加新 UI 的 dry-run 示例。
5. 执行全部 6 种方法、微调/评估分支的命令构造检查。

## 10. 验收标准

### 微调任务

- 可以选择三类扰动方法。
- 每类可以选择两个具体方法。
- 可以选择或确认模型。
- 可以选择或确认数据集。
- 可以选择当前方法或对比方法。
- 当前方法映射到 `utility`。
- 对比方法映射到 `baseline`。
- 未选择的参数全部从 JSON 读取。
- dry-run 能打印完整命令但不启动训练。
- 真实运行能写入日志、manifest 和训练指标。

### 评估任务

- 可以选择直接评估或间接评估。
- 可以选择显式微调模型或隐式微调模型。
- 显式分支支持梯度扰动、嵌入扰动、无扰动接口。
- 隐式分支支持示例扰动、无扰动接口。
- 梯度扰动和嵌入扰动各自可以继续选择两个具体方法。
- 无扰动只提示暂不支持，不启动命令。
- 直接评估映射到 `direct_attack`。
- 间接评估映射到 `indirect_attack`。
- 历史评估结果只展示当前方法对应文件。

### 兼容性

- `privacy_tool.py` 无需修改。
- 现有命令行 `list`、`doctor`、`run` 暂时可用。
- 原始六套研究代码不被改写。
- `smoke_test.py` 和 `full_audit.py` 能通过新的结构检查。

## 11. 当前不在范围内的内容

- 不修改六种方法的核心算法。
- 不重新设计训练脚本参数。
- 不实现无扰动评估。
- 不自动新增模型或数据集下载逻辑。
- 不删除旧命令行接口。
- 不改变历史结果文件内容。
