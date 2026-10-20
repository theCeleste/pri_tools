# Glm-subj 接入规划

本文规划将 `Glm-subj/` 中面向 ChatGLM3-6B / Subj 的微调与隐私评估实现接入统一工具。本文仅描述改动方案，不代表代码已完成接入。

## 1. 目标与边界

- 保留现有方法和模型路径；只在选择 ClipText、PrivSV、ZO-DPGA 时增加 ChatGLM3-6B / Subj 分支。
- 微调当前方法对应 Glm-subj 中的保护实现，效用对比对应配套的非保护实现。
- 性能对比保留 DP-SGD 入口，但当前不运行 DP-SGD 实验。
- ChatGLM3-6B 的直接评估接入 Glm-subj 的 `privacy_attack/`；间接评估仅留入口，后续补充。
- 微调与评估的历史结果分别从 `Glm-subj/outputs/` 下的训练和攻击结果读取，不与既有 Llama 或无扰动历史混用。

## 2. 现有统一入口与需要调整的边界

交互菜单、选择状态和分支路由集中在 [`unified.py`](unified.py)，方法命令构造集中在 [`catalog.py`](catalog.py)，配置示例在 [`settings.example.json`](settings.example.json)。

当前微调第三步已包含 ClipText、PrivSV、ZO-DPGA；改动重点是选择这些方法后增加模型和数据集选项，并将选择传给对应的运行器。当前 `_choose_model_and_dataset()` 只显示配置值，且会将模型覆盖为固定的 Llama 字符串；数据集只读取一个配置值。评估中，具体方法选完后也会进入旧方法路径；现有 ChatGLM 评估路由主要服务于“无扰动微调”，不能直接复用作这三种方法的评估路由。

建议在选择状态中保留用户可见的 `method_id`，并另设明确的 Glm-subj 后端标识。不要通过修改全局 Llama/SST-2 配置或替换原有方法 builder 来实现，以免影响现有流程。

## 3. 微调分支

### 3.1 选择流程

在第三步选择 ClipText、PrivSV 或 ZO-DPGA 后，模型选择增加 ChatGLM3-6B，数据集选择增加 Subj。保留原有选项；仅 `ChatGLM3-6B + Subj` 是本次新增支持组合，遇到不支持的交叉组合时应提示并阻止启动。

仓库配置已存在 `common.chatglm3_6b`，但当前选择器没有正确使用它；`common.dataset` 目前是单一配置值，因此应增加 Glm-subj 专用的数据路径与模型/输出配置，不要复用 SST-2 路径。

### 3.2 当前方法与效用对比映射

| 方法 | 当前方法（保护） | 效用对比（非保护） |
|---|---|---|
| ClipText / Embedding | ClipText+ Subj 数据 + causal LoRA | Raw Subj 数据 + causal LoRA |
| PrivSV / Example | Raw causal LoRA + Private Steering | 同一 Raw causal LoRA + Mean Steering |
| ZO-DPGA / Gradient | DPZero causal LoRA | MeZO causal LoRA |

这些配对与 [`Glm-subj/PROTECTION_AND_DIRECT_ATTACKS.md`](../Glm-subj/PROTECTION_AND_DIRECT_ATTACKS.md) 一致。ClipText+ 数据已准备在 `Glm-subj/data/subj_embedding_protected/`，原始数据位于 `Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10/`。

Glm-subj 的实现入口包括 [`train_lora.py`](../Glm-subj/privacy_attack/train_lora.py)、[`build_steering.py`](../Glm-subj/privacy_attack/build_steering.py) 和 [`train_zo.py`](../Glm-subj/privacy_attack/train_zo.py)。PrivSV 是基于 Raw LoRA 构造并在推理时注入 steering 的方法，不是另一份独立 LoRA 训练；其文档指出当前实现没有 epsilon accountant，结果不得描述为形式化 DP 保证。

### 3.3 性能对比占位

保留“性能对比方法：DP-SGD”选项和后端接口，但 ChatGLM + Subj 分支暂不调用已有 DPSGD 实现。选择该项时应明确提示“尚未接入”，不应返回旧模型的 DP-SGD 结果，也不应将占位运行记作成功实验。

### 3.4 效用指标待定义

Glm-subj 训练报告主要记录训练/验证 loss、perplexity、运行时间及隐私相关信息；现有代码没有统一的 Subj 分类准确率评估。统一工具当前的效用展示更偏向 accuracy。实现前需确定统一效用口径：建议使用固定提示模板在相同 Subj holdout 上评估分类准确率，并可同时记录 loss/perplexity。PrivSV 的 Mean/Private 两个 steering 分支也必须使用同一效用评估器，否则该方法的保护与基线效用不可直接比较。

## 4. 评估分支

### 4.1 选择与路由

评估流程当前依次选择直接/间接评估、微调方式、扰动类别和具体方法。选中 ClipText、PrivSV 或 ZO-DPGA 后，模型选择增加 ChatGLM3-6B；选中 ChatGLM3-6B 时数据集隐含为 Subj，不再展示独立数据集选择。现有评估方法类别布局保持不变（当前 PrivSV 位于隐式方法组，ClipText 和 ZO-DPGA 位于显式方法组）。

ChatGLM 分支应在进入现有 Llama builder 前路由至 Glm-subj 专用评估入口，并检查所选模型产物、数据路径和输出目录。现有 Llama 评估、无扰动 ChatGLM 评估均保持原行为。

### 4.2 直接评估

直接评估采用 Glm-subj 的 known-prefix causal extraction，即 [`prefix_attack.py`](../Glm-subj/privacy_attack/prefix_attack.py)；方法配对结果由 [`compare.py`](../Glm-subj/privacy_attack/compare.py) 汇总。建议沿用该目录定义的保护/非保护比较：

| 方法 | 保护侧 | 非保护对照 | 评估 reference |
|---|---|---|---|
| ClipText | ClipText+ adapter | Raw LoRA adapter | 原始 source 文本；不以扰动后的文本替代主结果 |
| PrivSV | Raw LoRA + Private Steering | 同一 Raw LoRA + Mean Steering | 原始 Subj 文本 |
| ZO-DPGA | DPZero adapter | MeZO adapter | 原始 Subj 文本 |

保护侧与对照侧应使用匹配的目标样本、相同攻击参数，并分别报告 train 与 holdout 指标及命中数/分母。`causal_mia.py` 是 loss-based membership inference；除非后续重新定义评估类型，第一阶段不把它归入“直接评估”。

评估所需 adapter / steering 路径应由 Glm-subj 专用设置提供，或引用微调分支生成的模型目录。缺少产物时应明确提示先完成对应训练/构建，不使用旧 Llama adapter 路径作为默认值。

### 4.3 间接评估占位

ChatGLM + Subj 的间接评估保留选项和后端接口，但本阶段不调用现有 Llama 邻居攻击，也不将其他模型的历史结果显示为当前结果。选择该项时给出“尚未接入”的明确状态；待后续确定适用的攻击实现、数据格式和指标后再启用。

## 5. 历史结果读取与输出组织

现有 [`core.py`](core.py) 的通用历史读取器按方法读取 `privacy_tools/historical_results` 下的静态文本；现有 [`evaluation_history.py`](evaluation_history.py) 服务于无扰动评估身份。这两套逻辑均不能直接代表 Glm-subj 的方法级训练和攻击历史。

### 5.1 输出约定

建议统一工具将每次 Glm-subj 运行写入 `Glm-subj/outputs/<run_id>/`，并通过现有 `MODEL_ROOT` / `ATTACK_ROOT` 参数约定分别放置模型和攻击产物：

```text
Glm-subj/outputs/<run_id>/
├── run_manifest.json
├── models/       # adapter、steering、training_report.json、build_report.json
└── attacks/      # prefix/summary.json、逐条 records、方法比较 JSON
```

在配置中提供 Glm-subj 输出根目录；运行 manifest 至少记录后端、方法、模型、数据集、实验类型、保护/基线角色、状态、时间以及训练报告和攻击结果路径。历史读取应按 manifest 身份筛选最近一次成功运行，不能只依赖目录名或文件修改时间。

### 5.2 微调历史

- `utility` 读取当前方法的保护侧训练/构建报告及效用评估结果。
- `baseline` 读取相同方法对应的非保护报告及同口径效用评估结果。
- LoRA / DPZero / MeZO 训练主要读取 `training_report.json`；PrivSV 还需读取 `build_report.json` 和统一效用评估产物。
- `performance_baseline` 在未实现前显示无可用结果，不回退到旧 DP-SGD 历史。

### 5.3 评估历史

- ChatGLM + Subj + 直接评估：按所选方法读取匹配的 `attacks/**/prefix/summary.json`，存在时同时读取相应的 comparison JSON；展示 train、holdout、差值及分母。
- ChatGLM + Subj + 间接评估：目前显示无可用历史/尚未接入，不读取旧方法历史。
- Llama 与无扰动评估继续使用其现有历史读取方式。

## 6. 配置与环境检查

在 [`settings.example.json`](settings.example.json) 中增加 Glm-subj 专用配置，覆盖 base model、Raw/Embedding 数据目录、输出根目录、保护/基线产物路径及 Python 解释器。当前 Python 环境检查按旧方法 ID 选环境；接入时需为新后端使用与 [`Glm-subj/requirements.txt`](../Glm-subj/requirements.txt) 相容的环境配置，并更新 [`bootstrap.py`](bootstrap.py) 的依赖检查。

模型与数据集 doctor 检查也需读取本次实际选择的 ChatGLM 路径和 Subj 路径，而不是继续检查 `common.model_7b` 或 `common.dataset_path`。

## 7. 实施顺序

1. 扩充选择状态和条件菜单，记录方法、模型、数据集及 Glm-subj 后端身份；校验仅支持的模型/数据组合。
2. 增加 Glm-subj 配置、Python 环境与路径检查。
3. 为三个方法分别接入保护微调和非保护效用对照 builder；为 DP-SGD 建占位接口。
4. 增加三种方法的直接评估 builder，固定保护/对照映射、reference 和匹配攻击目标；为间接评估保留无执行的占位。
5. 规范每次运行输出与 manifest，分别增加微调报告和评估 summary 的历史读取。
6. 补充选择路由、命令构造、历史筛选和旧路径兼容测试，并用 dry-run 验证命令、数据及产物路径。

## 8. 验收要点

- 只有选择 ClipText、PrivSV、ZO-DPGA 时才出现 ChatGLM3-6B / Subj 相关选项；现有模型路径不回归。
- 三种方法的保护实现、非保护效用基线、直接评估模型和 reference 均与上表一致。
- ChatGLM + Subj 不会误调用旧 Llama 训练、DP-SGD、直接攻击或邻居攻击实现。
- 训练与评估历史各自从 Glm-subj 输出中读取，并按方法、角色、模型、数据集、成功状态过滤。
- 缺失训练/攻击产物时给出清楚提示；占位的 DP-SGD 和间接评估不会伪造成功结果。
- 所有新流程均有 dry-run 命令预览和针对路由/结果读取的自动化测试。
