# ChatGLM / FM9G 间接评估统一替换实施规划

日期：2026-09-27  
状态：接入代码已实施，CPU 接入与离线重算验收通过；真实 GPU 评估待资源和数据前置条件满足后验收。下文第 1–9 节保留原规划记录，当前完成情况见第 10 节。

## 1. 结论与范围

**本方案能够同时覆盖两个目标：替换 ChatGLM 现有间接评估方法，并实现九格 FM9G-7B 预留的显式、隐式微调间接评估接口。**

可行性依据是 `evaluate/` 的两套后端已经支持 ChatGLM3-6B 与 FM9G-7B；待实现的工作主要是统一路由、外部资源参数、数据校验和结果管理。真实运行仍需相应底座、适配器、邻居模型和依赖环境，并通过下述验收。

| 菜单模型 | 微调方式 | 当前入口状态 | 替换后的唯一计算后端 | 后端模型标识 |
| --- | --- | --- | --- | --- |
| ChatGLM3-6B | 显式 | 调用 `glm/GLM_LoRA` | `evaluate/indirect_LoRA` | `chatglm3` |
| ChatGLM3-6B | 隐式 | 调用 `glm/GLM_ICL` | `evaluate/indirect_ICL` | `chatglm3` |
| FM9G-7B（九格） | 显式 | 提示预留并提前返回 | `evaluate/indirect_LoRA` | `fm9g` |
| FM9G-7B（九格） | 隐式 | 提示预留并提前返回 | `evaluate/indirect_ICL` | `fm9g` |

统一入口停止调用旧 `glm/` 评估代码，不设置自动回退。旧目录和历史结果保留。直接评估的预留状态、其他扰动方法及训练流程不在本次替换范围内；不自动重训 LoRA，也不执行未微调模型对比。

## 2. 已核对的源码依据与缺口

- `privacy_tools/unified.py`：ChatGLM 两个间接评估处理函数仍引用旧目录；FM9G 在 `interactive()` 中提前返回；无扰动选择仍借用 `kal_dp` / `dadp_icl` 后端身份。
- `evaluate/indirect_LoRA/run.py`：`MODELS` 包含 `chatglm3`、`fm9g`，`DATASETS` 包含 `sst2`、`wikitext2`；支持 attack、evaluate、check。适配器、邻居模型与数据目录目前固定在包内。
- `evaluate/indirect_LoRA/code/attack.py`：已支持 `--dataset-file` 与 `--dataset-dir`，可复用 TSV 加载和现有打分算法。
- `evaluate/indirect_ICL/DPICL/runtime_config.py`：已有两个底座模型别名及 `CHATGLM_MODEL_DIR` / `FM9G_MODEL_DIR`。
- `evaluate/indirect_ICL/DPICL/scripts/run_indirect.py`：已有 `--target-model`、`--proposal-model`、`--sst2-dir`、`--task-dataset`、`--wikitext2-dir`、`--input-json` 等参数。
- 两个包均有 `METHOD.md`、测试和完整性校验；支持上述模型不等于当前机器的权重文件及依赖已经齐全。

## 3. 接入架构

### 3.1 路由与任务身份

在 `unified.py` 中按“评估类型 → 模型 → 微调方式”分派：

1. 间接评估且模型为 ChatGLM 或 FM9G，进入新的通用间接评估处理器。
2. `explicit` 选择 LoRA builder；`implicit` 选择 ICL builder。
3. 仅对 FM9G 间接评估移除提前返回；直接评估继续显示预留提示。
4. 规范化菜单标识 `chatglm` 为后端 `chatglm3`；拒绝未知组合，不回退到 Llama、Kal-DP 或 DADP。
5. 在访问借用的 `METHODS[method_id]` 之前分流，避免沿用错误解释器、资源检查和历史结果身份。

建议新增 `privacy_tools/indirect_evaluation.py`，集中配置解析、命令构造、运行管理及摘要处理；由 `catalog.py` 按需导出 builder，`unified.py` 仅负责交互。实际组织方式可服从现有框架，但不得在交互层复制两套攻击算法。

### 3.2 LoRA 流程

调用新版 `evaluate/indirect_LoRA/run.py`：

```text
加载指定底座 + 匹配的 LoRA
→ member 打分 → non-member 打分
→ 有效分数校验 → 原生 evaluate.py → 统一结果摘要
```

对新版入口增加最小外部参数：adapter、邻居模型、成对 member/non-member 文件，以及需要时的数据目录、tokenizer 路径。保持原包默认行为兼容；不修改其核心损失、邻居生成或阈值算法。参数应同时用于实际子命令、资源预检与运行清单，不能仅出现在配置中。

### 3.3 ICL 流程

统一层调用新版 Python 编排入口并显式设置运行环境，复用其现有路径参数：

```text
准备数据 → 生成示例（或读取已有示例）
→ original_text：成员/非成员攻击 → 融合 → 评估
→ synthetic_text：成员/非成员攻击 → 融合 → 评估
→ 统一结果摘要
```

离线重算调用新版原生 combine 和 evaluate 脚本，或封装其已有 evaluate 模式；不加载目标模型。隐式评估只加载底座，不加载 LoRA。

## 4. 配置、模型与环境

采用共享后端配置、模型独立资源映射，避免维护 ChatGLM 和 FM9G 的重复实现。建议使用 `evaluation.indirect` 配置域；兼容读取旧 ChatGLM 配置时，只转换为新后端参数，不调用旧代码。

| 配置项 | 规划 |
| --- | --- |
| 默认数据集 | SST-2；读取项目配置，不在后端固定 |
| SST-2 目录 | `common.dataset_path`，默认 `privacy_tools/datasets/sst2/` |
| ChatGLM 底座 | 环境变量 `CHATGLM_MODEL_DIR` > 配置中的本地底座路径 > 明示包默认路径 |
| FM9G 底座 | 环境变量 `FM9G_MODEL_DIR` > 模型独立配置 > 明示包默认路径 |
| ChatGLM SST-2 adapter | 延续 `/home/zjs/llm/glm-lora/` |
| 其他 adapter | 按模型、数据集分别配置；未配置时检查新版 `weights/{model}/{dataset}/` |
| LoRA 邻居模型 | 延续 `/home/zjs/llm/roberta-base/` |
| ICL proposal 模型 | 新版 `evaluate/indirect_ICL/resources/bert-base-uncased/`，允许外部 BERT 路径覆盖 |
| Python | 为 LoRA、ICL 分别提供可配置解释器，不强制使用 UI 进程的 `sys.executable` |

明确本地模型目录契约，不将 `THUDM/chatglm3-6b` 之类仓库标识拼为项目内路径，也不静默联网下载。配置缺失时提供具体配置键及环境变量提示。

ICL 沿用新版 BERT 提议、文本清理和 embedding dropout 方法。旧配置中指向 RoBERTa 的 `common.icl_proposal_model` 不应被无提示继承；迁移时明确提示改用 BERT。

GPU 使用同一映射约定：物理卡通过 `CUDA_VISIBLE_DEVICES` 限定，子进程使用逻辑 `cuda:0`。LoRA 原入口已有该映射；ICL 直调 Python 时由统一层负责。多卡字符串、不可用卡号应明确校验，不重复映射。遵循原包依赖和精度设置，允许两套独立环境；不自动安装或更改精度。

## 5. 数据一致性与方法口径

### 5.1 SST-2 使用统一路径

继续遵守已确定的项目统一 SST-2 路径要求，不复制、覆盖统一数据文件。

- ICL：train 为私有示例采样池、dev 为分类效用数据、test 为非成员池；成员来自生成记录的 `original_text`，不是直接把 train 前十条标为成员。
- LoRA：通过新增文件参数读取统一路径；必须确认选取的成员确实属于对应 adapter 的训练数据。不能仅凭 `train.tsv` 文件名确立成员身份。
- 统一 test.tsv 实际为 `index、sentence`，不要求缺失的情感标签；攻击只评估正文。
- 对选取样本记录数量、顺序、来源和指纹，按新版归一化规则检查重复、交集和空文本。
- adapter 训练划分需有可核对的索引、指纹或来源记录。若统一数据中的候选成员不属于训练划分，或来源无法核实，明确阻止正式 MIA 结果生成并报告原因；不得静默换用包内数据、重新定义成员或重训模型。命令预览可列明该待解决项。

### 5.2 数据集扩展

后端参数保留 SST-2 / WikiText-2 支持，默认 SST-2，不在本次任务中强制增加菜单层级。WikiText-2 使用独立数据源配置：LoRA 的 HF 固定划分与 ICL 的 JSON 数据格式分别映射，不能把一个目录同时当作两种格式。没有显式路径时可使用新版对应快照，并记录实际路径。

### 5.3 保留新版原生方法

| 项目 | LoRA | ICL |
| --- | --- | --- |
| 权重更新 | 加载已有 adapter，不重训 | 无权重更新 |
| 默认成员 / 非成员数 | 1200 / 1200 | 10 / 500 |
| seed | 42 | 0 |
| 邻居方法 | RoBERTa mask，top-k=8，最多100邻居 | BERT embedding dropout=0.7，top-k=6，前10邻居评分 |
| 提示与分数 | 正文及 EOS 的平均 log probability 差值 | 排除提示及 padding 的正文平均 log probability 差值 |
| 结果分组 | 单模型 | original_text / synthetic_text 分开 |
| 精度 | BF16 | 生成 FP16、攻击 FP32 |

两类均保留原生 AUC 与各目标 FPR 下的 TPR，不统一改写计算器，不插值或伪造不可达 FPR。ICL 的 `num-test=20` 是 SST-2 分类效用测试数，不是攻击样本数；WikiText-2 不做情感分类效用测试。ICL 默认无生成噪声，不应将合成示例自动标注为有差分隐私保证的保护模型。LoRA 权重训练成员和 ICL 上下文成员含义不同，摘要应区分。

## 6. 执行模式与结果管理

统一入口区分三种行为；命令预览仅作为底层模块的诊断能力，不在无扰动评估交互菜单中提供：

1. 立即执行：运行选定新版后端，失败返回非零状态并显示日志位置。
2. 查看最近历史结果：只读查询模型、微调方式、数据集、后端版本匹配的成功结果，不加载模型、不创建任务。
3. 离线重算：仅读取已有分数/攻击 CSV，核对来源元数据，输出到新目录。

底层模块仍可使用 `--mode preview` 打印具体阶段命令及必要环境映射，不启动计算、不创建输出；允许提示资源缺失而不假装已验证可运行。

建议目录：

```text
privacy_tools/results/indirect_evaluation/
  {chatglm3|fm9g}/{explicit|implicit}/{sst2|wikitext2}/{run_id}/
    run_manifest.json
    summary.json
    backend/  # 由原生入口创建，保留原生输出结构
```

父执行器可创建外层目录，但必须把尚不存在的 `backend/` 传给原生入口，避免与禁止覆盖检查冲突。运行 ID 使用微秒或 UUID 防止撞名。

保留 LoRA 的 score pickle、metadata、CSV，ICL 的 generated、两组 CSV、metrics.json、ROC 文件，以及所有阶段日志。统一 summary 记录模型、微调方式、数据集、后端版本、实际路径、数据来源、实际样本量、指标、原生结果位置和完成状态。数值从真实产物提取，不以默认样本数冒充实测数。

旧 `glm/` 结果只作为带 legacy 标识的历史记录，不与新后端结果自动混用。离线重算未知来源的旧结果需显式导入及来源标识，不宣称具有新版实验可比性。

## 7. 预计修改清单与实施顺序

| 阶段 | 文件 / 模块 | 交付与验证 |
| --- | --- | --- |
| 1 | `settings.example.json`、新适配模块 | 模型映射、资源配置、解释器及输入契约；纯命令构造测试 |
| 2 | `evaluate/indirect_LoRA/run.py` 及测试 | 外部 adapter / 邻居 / 数据输入，保持原默认模式兼容 |
| 3 | 新适配模块、`catalog.py` | LoRA / ICL builder、预检、GPU 环境、原生离线重算；优先不修改 ICL 算法 |
| 4 | `unified.py` | ChatGLM 路由替换；FM9G 两个间接接口接通；直接评估继续预留 |
| 5 | 适配模块，必要时 `core.py` 的局部扩展 | 历史查询、命令预览、运行目录与摘要，避免改变其他方法语义 |
| 6 | `smoke_test.py`、独立集成测试、README | 四条用户路径回归、新版引用检查、配置与运行说明 |
| 7 | 新版包校验清单、真实环境 | 如修改包内文件，同步维护清单；完成 CPU 与 GPU 分层验收 |

只在确有必要时调整新版 ICL 入口参数，不复制其实现。修改交付包内文件时更新相应 checksums 清单，保留校验机制；外部配置资源另做运行预检，不把当前自带资源缺失误归因为适配器错误。工作区已有的其他改动和字节码产物不在本次清理范围。

## 8. 验收标准

### A. 确认两个目标已实现的必要条件

- ChatGLM 显式、隐式间接评估产生的全部计算命令只引用 `evaluate/` 新后端。
- FM9G 显式、隐式间接评估进入同一套新版流程，不再仅打印“待后续实现”，且不错误调用 ChatGLM 模型。
- 模型标识、底座、adapter、tokenizer 和数据集匹配；ICL 不携带 adapter。
- 两种模型的直接评估仍保持预留；既有其他方法的执行路径不受影响。

### B. 自动化与资源检查

- 纯命令构造测试覆盖 2模型 × 2微调方式 × 2数据集；mock 子进程验证路由、解释器、参数和 GPU 映射，不依赖真实 GPU。
- 菜单 smoke test 覆盖四个间接评估组合、取消操作、历史结果存在/缺失及直接评估占位。
- 预览不创建目录；历史查看不调用 subprocess；已有输出不覆盖；子进程失败不会输出成功摘要。
- 针对缺少底座、adapter、tokenizer、邻居权重、样本不足、划分重叠和配置类型错误提供准确提示。
- 合成分数 CPU 测试覆盖原生指标复算、两组 ICL 结果和 summary 数值一致性；离线模式不检查 GPU。
- 运行新版原有测试；区分纯逻辑测试与依赖模型/数据快照的交付测试。缺资源时明确列出未验收项，不能把 mock 通过当作真实模型通过。

### C. 实际运行验收

在资源和成员来源确认后，依次验证四条模型/微调方式链路，并以约定正式样本规模生成结果。若使用小样本开发冒烟运行，单独标记，不混入正式历史结果；不得自动缩减正式样本规模。核对原生产物、摘要、日志及可复算性。

**完成定义：两条 ChatGLM 间接评估链路替换成功，两条 FM9G 预留间接评估链路实际接通，四条路径均满足对应验收标准；未完成真实环境验收时明确标记，不宣称 GPU 评估已经通过。**

## 9. 本次文档交付边界

本次只新增本规划文档。已复核的是源码支持及接入可行性，未改动评估、菜单或配置代码，未安装依赖、加载模型、运行测试或启动评估。实施时首先检查真实资源路径与 adapter 训练划分记录；缺失项是正式运行前置条件，不改变本方案覆盖上述两项目标的结论。


## 10. 实施记录（2026-09-27）

- ChatGLM、FM9G 的显式与隐式间接评估均经统一处理器进入 evaluate/；在旧方法身份解析之前分流。直接评估继续预留。
- 新增 privacy_tools/indirect_evaluation.py 与 indirect_data_check.py，负责资源配置、数据来源预检、独立解释器、运行清单、日志、实际指标摘要、历史查询和离线重算。无需调整其他方法的 catalog/core 执行机制。
- LoRA 原生入口增加外部 adapter、tokenizer、邻居模型和成对数据文件参数；原算法及包内默认调用方式保持兼容。ICL 直接复用新版编排入口。
- 配置默认采用 /home/zjs/llm/glm-lora、/home/zjs/llm/roberta-base 和项目统一 SST-2 路径；ICL 使用新版 BERT proposal。底座与独立解释器需按安装位置配置。
- 更新菜单、smoke 检查、配置示例、使用说明及 LoRA 修改文件的校验清单。旧目录、旧结果和已有其他工作区改动保留。

### 已完成验证

- 新增接入测试与真实 CPU 离线子进程测试共 16 项通过；覆盖 2模型×2方式×2数据集命令、四个菜单入口、只读预览/历史、资源与成员来源错误、失败状态、四条路线的原生指标重算和摘要一致性。
- 合成分数通过真实原生评估器重算，四条路线结果 AUC=0.75；这些仅是测试夹具，临时目录自动清理，不代表任何模型的实验指标。
- 原生 ICL 测试共 18 项：17 通过、1 因外部模型/tokenizer 未安装跳过。
- 原生 LoRA 命令、数据不交叠与指标复算测试通过；adapter/tokenizer 测试中 FM9G 两个数据集因当前环境缺 datamodel_code_generator、jsonschema 报错，尚未全量通过。
- 只读数据校验：ICL 的统一 SST-2 与随包 WikiText-2 通过；当前统一 SST-2 前1200条候选成员并非全部属于默认 ChatGLM adapter 的训练集，显式运行会拒绝这种错配。

### 尚未完成的真实环境验收

真实 GPU 攻击未启动。当前机器需要补齐底座与缺失权重，为两套后端配置符合原生依赖版本的 Python 环境，并核实 LoRA 的完整训练来源及评估划分。本次没有下载/安装依赖、改变统一数据、自动重采样或重新训练 adapter。

使用与配置说明见 [INDIRECT_EVALUATION.md](INDIRECT_EVALUATION.md)。四条链路的接入与 CPU 层已验证；真实模型层通过前，不宣称 GPU 实验已经完成。
