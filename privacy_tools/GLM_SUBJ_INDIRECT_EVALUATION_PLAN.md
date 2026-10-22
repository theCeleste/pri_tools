# Glm-subj 间接评估接口接入规划

日期：2026-10-21  
状态：接入已实施；CPU 集成、原包完整性校验及三方法六变体 GPU 开发冒烟通过。完整样本规模正式评估尚未运行。

## 1. 目标与范围

利用 `Glm-subj/Indirect_attack/` 的实现，接通统一工具中 ClipText、PrivSV、ZO-DPGA 的 **ChatGLM3-6B + Subj + 间接评估** 预留接口。复用原生邻居生成、模型打分、固定样本选择与指标计算，统一层负责路由、配置、执行、摘要和历史查询。

本次接入范围是这三个方法的 Glm-subj 分支。现有无扰动 ChatGLM / FM9G 的 SST-2、WikiText-2 间接评估仍使用 `evaluate/indirect_LoRA`、`evaluate/indirect_ICL`；既有 Llama、直接评估、训练和 DP-SGD 流程保持各自路径。

首期默认执行交付包配套权重、数据及 steering 构造的复现实验，显示保护与非保护两组结果。它们不自动代表统一工具最近一次微调产物的风险；如需评估外部训练产物，须显式配置并通过对应的数据来源校验。

## 2. 已核对的接口现状

| 位置 | 当前行为 | 接入时的调整 |
| --- | --- | --- |
| `privacy_tools/unified.py` 的 `interactive()` | 已将三种方法的 ChatGLM3-6B / Subj 选择交给 `glm_subj.run()`，提供立即执行、历史查询、命令预览 | 复用现有入口，补充间接评估的资源说明与路径展示 |
| `privacy_tools/glm_subj.py` 的 `PLACEHOLDERS` | 包含 `indirect_attack`；执行仅提示尚未接入，构造步骤抛出异常，历史直接返回空 | 移除该占位，接入专用 builder、校验和摘要 |
| `identity()` | 仅直接评估被标记为 `role=paired` | 间接评估也使用 `paired`，另记录实际后端和资源来源 |
| `doctor()` | 面向 `privacy_attack` 的 `--base-model`、`--data-dir` 等参数 | 为间接评估分派专用预检，避免错误套用直接评估校验 |
| `runtime_profile()` / `bootstrap.py` | 默认使用 Glm-subj 训练环境和依赖清单 | 增加间接评估独立环境配置，指向新包依赖 |
| `Glm-subj/Indirect_attack/run.py` | 支持 `check/full/attack/evaluate`、方法、变体、底座、GPU、输出、邻居、resume 和 dry-run | 首期直接复用；外部资源覆盖只作必要参数扩展 |

统一 CLI 已有 `--backend glm_subj`，应与交互菜单进入相同适配逻辑，不再增加一套平行执行流程。

## 3. 方法和实验口径

| 工具方法 | 原生 `--method` | 非保护变体 | 保护变体 | 最终 member / nonmember | 目标打分长度 |
| --- | --- | --- | --- | --- | --- |
| PrivSV | `example` | `mean` | `private` | 1000 / 1000 | 最大 512 token |
| ClipText | `embedding` | `raw` | `cliptext_lora` | 100 / 1000 | 最大 512 token，prefix=0 |
| ZO-DPGA | `gradient` | `mezo` | `dpzero` | 100 / 1000 | 最大 14 token，prefix=8 |

三个方法均使用 BERT base uncased 提议邻居，embedding dropout=0.7、每位置 top-k=6、候选最多 50、实际评分前 10 个邻居。以原文平均 log probability 减去邻居平均 log probability 作为分数，分数越高越倾向 member。保留各方法原有提示、特殊 token 和 loss mask 处理，不以统一打分器重写。

PrivSV 先从 8000 条构造记录中按 seed=42 选择 1000 条，这些样本同时作为 member；攻击 seed=43。先攻击 1000 member + 2000 nonmember，再由 `select_subset.py` 按 seed=60 选择 1000 nonmember，两个变体共用样本与邻居。保留注入层 11–15、读取位置 -2、clip=18、noise multiplier=0.005、steering multiplier=256、private seed=42。

**PrivSV 的重要边界：**新包在 ChatGLM 底座上运行时构造 Mean / Private Steering，不加载 Raw LoRA，也不读取统一训练接口生成的 steering 目录；现有直接评估则使用 Raw LoRA + 已保存 steering。此次接入按新包方法执行，在任务确认和结果中注明实际模型组成，不沿用直接评估的产物配对规则。Private 的噪声参数本身不构成已验证的 epsilon 保证。

## 4. 资源选择与训练产物关系

建议增加 `glm_subj.indirect` 专用配置域，避免间接评估误读已有 `glm_subj.artifacts` 中直接评估使用的权重。

| 配置项（规划） | 默认值或含义 |
| --- | --- |
| `python.glm_subj_indirect` | 新包兼容的 Python 解释器；未配置时可回退 `python.glm_subj`，但仍按新包依赖检查 |
| `glm_subj.indirect.resource_source` | `bundled`；另支持显式选择 `external` |
| `glm_subj.base_model` | 沿用现有 ChatGLM 本地底座配置，并显式传入 `--model` |
| `glm_subj.indirect.proposal_model` | `Glm-subj/Indirect_attack/resources/bert-base-uncased` |
| `glm_subj.indirect.embedding` | `baseline_adapter`、`protected_adapter`、`splits`；默认包内 `embedding/weights/{raw,cliptext_lora}` 与 `embedding/data/splits` |
| `glm_subj.indirect.gradient` | 对应 `gradient/weights/{mezo,dpzero}` 与 `gradient/data/splits` |
| `glm_subj.indirect.example.interface_dir` | 默认 `example/interface`，包含构造数据、测试数据和配套接口代码 |
| `glm_subj.indirect.neighbors` | 可选已有邻居缓存；必须核对方法、样本与生成参数 |
| `common.gpu` | 单张非负物理 GPU 编号 |
| `glm_subj.output_root` | 沿用 `Glm-subj/outputs`，每次生成独立运行目录 |

所有项目相对路径先基于项目根解析为绝对路径，再传给原生入口；不依赖 UI 启动目录或原包的 `../models` 默认路径。统一传入配置后的底座，避免 shell 环境变量造成不可见覆盖。

### 4.1 随包资源模式

默认使用新包提供的成对权重与固定划分，不调用 `artifact_pair()` 自动选择最近训练模型。任务确认明确展示“随包权重复现”，记录权重和数据指纹、样本来源与方法标识。对成员来源的说明限于交付包的方法文档和实际可核查记录，不将文件存在或 checksum 通过表述为独立验证了完整训练来源。

### 4.2 外部资源模式

外部 embedding / gradient 评估必须成对指定 adapter、共同攻击划分及训练来源记录；不能只替换权重并继续把随包 train 文件视为 member。校验正文/原始样本 ID 与对应训练记录的映射，以及 nonmember 未进入训练数据；ClipText 的保护文本需保留与原始文本的映射证据。

当前工具配置中的训练超参与新包不同：例如现有 ZO 配置 rank=64、max_steps=6750，而新包描述 rank=1、9000 步；现有 LoRA 默认训练 1 epoch，新包描述为 15 epoch。因此不能根据同一方法名称推定权重和划分可替换。

为支持显式覆盖，规划对原生 `run.py` 最小增加成对 adapter、splits、proposal model、example interface-dir 参数，保持原默认值和算法。example 中攻击样本加载与 `create_subj_models()` 的构造数据必须使用同一接口目录/数据文件；外部路径扩展需显式传入构造数据路径，避免只改攻击输入却仍用默认数据构造向量。

外部产物来源无法确认时，预览说明缺失项，正式执行返回明确错误。首期不自动重训、不自动替换数据，也不把已保存的直接评估 steering 作为新包 example 的输入。加载该类 steering 或 Raw LoRA + Steering 属于后续独立方法扩展。

## 5. 接入结构与执行流程

建议新增 `privacy_tools/glm_subj_indirect.py`，集中方法映射、命令构造、专用预检和结果汇总；`glm_subj.py` 继续管理运行目录、日志、确认、manifest 及历史入口。

默认构造一条原生编排命令，例如 ClipText：

```text
{python.glm_subj_indirect} {project}/Glm-subj/Indirect_attack/run.py
  --mode full --method embedding --model {absolute_base_model}
  --gpu {physical_gpu} --output {absolute_run_dir}/indirect
```

工作目录固定为 `Glm-subj/Indirect_attack`。直接调用 Python 入口，确保内部 `sys.executable` 继续使用同一解释器，避免 `run.sh` 的 PATH 中 `python` 选错环境。

| 方法 | 原生完整流程 |
| --- | --- |
| embedding / gradient | 生成一次邻居 → baseline 攻击及原生指标计算 → protected 攻击及原生指标计算 |
| example | 构造 Mean / Private 向量 → 共用邻居完成两组完整池攻击 → 固定子集选择 → 两组 `evaluate.py` |

`full` 当前强制运行两个变体，即使传入单个 `--variant`。默认成对评估直接使用这一行为，不向用户提供实际无效的单变体 full 参数。原生 `attack` 支持指定变体与缓存；`evaluate` 可离线计算指标。首期统一菜单沿用立即执行、历史、预览三种模式，不新增离线重算菜单；原生阶段模式作为诊断能力保留。

配置了邻居复用时，需区分原生行为：example 的 full 会使用传入缓存；embedding / gradient 的 full 会重新生成邻居，不使用 `--neighbors`。后两者应改走 `attack --variant both --neighbors ...`，并补足指标与配对校验，不能声称 full 已复用缓存。

## 6. 环境、预检与运行保障

1. 在 `bootstrap.py` 增加 `glm_subj_indirect` profile，requirements 指向 `Glm-subj/Indirect_attack/requirements.txt`。检查 torch、transformers、peft、accelerate、numpy、pandas、sklearn、sentencepiece 等实际依赖；参考新包固定版本和 README 的 Python 3.10 / PyTorch 2.5.1 建议，不将训练环境的 opacus 检查强加给间接评估。
2. 对实际选中方法检查底座配置与分片、tokenizer、adapter 配置及权重、BERT、数据格式、空文本、重复、划分交集与预期条数。包内 `verify.py` 同时检查全部方法资源和 checksum，适合完整交付验收；运行时仍需选中方法专用检查，避免因未选方法缺文件而误阻断。
3. 新包由 `--gpu` 设置 `CUDA_VISIBLE_DEVICES`，内部使用 `cuda:0`。统一层检查同一物理卡，拒绝负数或多卡字符串，不叠加第二套编号重映射。
4. 预览仅构造和展示命令、cwd、GPU 映射、权重与数据来源，不创建目录、不加载模型、不检查 GPU、不安装依赖。历史查询只读 manifest 和结果，不要求原底座或训练权重仍在本机。
5. 每次正式运行使用现有时间戳 + UUID 目录。原生入口使用 `mkdir(exist_ok=True)`，不能依赖它防覆盖；统一层应保证新输出路径并拒绝复用已有任务目录，首期不自动开启 `--resume`。
6. 检查子进程退出码、两组完整 scores 和 metrics、有限数值、实际数量、失败记录及配对一致性，全部满足后才能标记 success。失败保留日志与部分产物，历史不选入失败任务。
7. 若扩展包内源码，同步维护 `checksums.json`，保留包校验能力；测试禁写字节码，缓存和临时结果放到包外，避免 `verify.py` 将其视为交付污染。

## 7. 结果、摘要与历史

沿用现有 `output_root/*/run_manifest.json` 一级扫描契约，不改变其他实验历史布局：

```text
Glm-subj/outputs/{timestamp_uuid}/
  run_manifest.json
  run.log
  indirect_summary.json
  indirect/
    neighbors.jsonl                 # embedding / gradient
    full_pool/                      # example 的完整攻击池
    {baseline_variant}/scores.csv
    {baseline_variant}/evaluation/metrics.json
    {baseline_variant}/evaluation/metrics.csv
    {protected_variant}/scores.csv
    {protected_variant}/evaluation/metrics.json
    {protected_variant}/evaluation/metrics.csv
```

保留原生 `run_config.json`、失败记录、邻居元数据、固定子集记录等实际产物。`artifacts` 至少登记两组 metrics 和统一 summary，`products` 不宣称产生新的微调 adapter。

manifest 增加 `evaluation_backend=glm_subj_indirect`、schema 版本、resource_source、变体映射、实际底座/权重/数据路径及指纹、GPU、解释器、方法参数和原生结果路径。身份仍为 `backend=glm_subj`、各自 method、`model=chatglm3-6b`、`dataset=subj`、`experiment=indirect_attack`、`role=paired`。

统一 summary 读取原生 `metrics.json`，分别展示 AUC、实际 member/nonmember 数量及 TPR@FPR。保留目标 FPR、实际 FPR、阈值与混淆计数；可显示 protected - baseline 差值，注明方向，不把风险下降自动表述为效用提升或形式化隐私保证。

原生目标 FPR 为 100%、50%、10%、1%、0.1%、0.01%，采用严格 `score > threshold`，在实际 FPR 不超过目标值时选择最大 TPR，不插值。1000 nonmember 的 FPR 步长为 0.1%，0.01% 对应零假阳性区域，应同时呈现实际 FPR。

历史按身份、成功状态、后端版本及资源模式筛选，依据记录的完成时间选择最近结果。若配置了外部权重，应匹配记录的实验来源，避免把随包结果显示成当前外部模型结果。没有匹配结果时显示“尚无成功历史”，不回退到其他方法、数据集、无扰动评估或直接评估历史。

## 8. 预计修改清单与实施顺序

| 阶段 | 文件 | 交付内容 |
| --- | --- | --- |
| 1 | 新增 `privacy_tools/glm_subj_indirect.py` | 三方法映射、随包资源解析、纯命令构造、专用预检与结果契约 |
| 2 | `bootstrap.py`、`settings.example.json` | 独立解释器与依赖 profile、indirect 配置示例及默认资源模式 |
| 3 | `glm_subj.py`、必要的 `unified.py` 路径展示 | 移除占位，接入 builder/doctor，paired 身份、摘要及历史过滤 |
| 4 | 原生 `run.py`、必要的 example 数据路径传递、`checksums.json` | 最小外部资源覆盖，保持随包默认调用兼容，校验外部来源 |
| 5 | 新增 `tests/test_glm_subj_indirect.py`，调整 `tests/test_glm_subj.py` | 路由、命令、资源与结果校验、历史和失败行为测试；移除间接评估占位断言 |
| 6 | `privacy_tools/README.md`、`GLM_SUBJ_INTEGRATION_PLAN.md` | 更新当前接口状态、资源模式、方法差异与结果说明 |
| 7 | 原生测试、CPU 集成验证、真实 GPU 验收 | 按下述标准分层验证，记录实际完成层级 |

## 9. 验收标准

### 9.1 CPU 与不加载模型的验证

- 三个菜单入口和 CLI 均进入 `Glm-subj/Indirect_attack`，底座、方法、解释器、GPU、cwd 和输出目录正确，不调用旧 Llama 或无扰动后端。
- 成对变体映射、example 固定子集、gradient 的 14/8 打分参数保持原生口径；PrivSV 不错误加载直接评估 Raw LoRA 或 steering。
- 预览无写入，历史无子进程和 GPU 检查；不同方法、来源模式、失败状态和旧后端结果互不混用。
- 缺权重、底座、BERT、样本不足、划分重叠、来源无法核实、配置无效和子进程失败时给出具体错误，不能形成成功记录。
- 用临时合成 scores 调用真实原生 `evaluate.py`，核对 AUC、目标/实际 FPR、阈值、两组 summary、错误样本数和 NaN/Inf 拒绝行为。测试结果不写入正式历史。
- 配对检查覆盖两组样本 ID、正文和邻居不一致；example 子集 seed 一致。仅数量一致不足以视为匹配。
- 执行原包 `tests/test_delivery.py` 和现有 Glm-subj 相关回归测试；区分逻辑通过与真实资源验收。

### 9.2 真实模型验收

资源就绪后分别执行 embedding、gradient、example 完整流程，检查两个变体均产生可复算的分数、指标、统一摘要及成功 manifest；再次打开统一工具可读到相同身份的最近结果。

先做开发冒烟时单独标记，不能自动缩减正式样本规模后写成正式结果。正式验收使用上表完整样本数量；外部产物模式另需训练来源与攻击划分验收。

完成定义：三个预留间接评估接口具备真实执行、命令预览和匹配历史读取能力，保护/基线两组结果与原生实现一致；未执行真实 GPU 评估时只声明接入或 CPU 验证完成。

## 10. 初始规划交付说明

本次仅新增此规划文档。已静态核对统一接口、新包入口、方法说明、配置、模型加载和现有测试；未修改实现或配置，未安装依赖、运行测试、加载模型或启动评估。工作区已有改动不在本次处理范围内。


## 11. 实施记录（2026-10-21）

### 已实现

- 新增 `privacy_tools/glm_subj_indirect.py`，复用原包编排和指标计算；三个方法的交互/CLI 预留入口均接通，支持立即执行、只读历史和无写入预览。
- `glm_subj.py` 移除间接评估占位，使用 paired 身份、专用 doctor、资源快照、SHA256、实际解释器与 GPU 记录；沿用现有日志和失败 manifest 管理。
- 新增 `glm_subj_indirect` 依赖 profile 和 `python.glm_subj_indirect` 配置，未配置解释器时回退 Glm-subj 解释器；不要求间接攻击环境安装 opacus。
- 默认随包资源与直接评估产物隔离；外部 adapter 必须成对配置并附与权重 SHA256 绑定的原始训练正文来源记录，验证 member/nonmember；外部 example 显式提供接口目录。
- 原生入口新增 proposal、两侧 adapter、splits、interface-dir 参数。example 构造向量显式读取同一 interface-dir 下的数据，修正外部接口目录只改变攻击样本的隐患；保留原生默认调用方式。
- 邻居复用校验 ID、正文、数量、提议模型路径和生成参数；adapter 方法使用 attack 模式复用。正式分数校验来源、连续索引、有限数值、样本数、配对、AUC 和阈值对应的混淆计数，完整后才生成成功摘要。
- 原包完整性清单同步更新；README、路径总览和原集成规划同步更新。保留工作区原有配置变更和字节码文件。

### 已通过的验证

1. `/home/zjs/anaconda3/envs/mia/bin/python -B -m unittest privacy_tools.tests.test_glm_subj_indirect privacy_tools.tests.test_glm_subj privacy_tools.tests.test_glm_dpsgd -v`：31 项测试全部通过；其中新增集成测试适配调用原包四项交付测试，无需安装 pytest。
2. CPU 测试通过真实原生 `evaluate.py` 对三种方法的正式数量合成分数重算指标；同时验证错误分数、缺失/不匹配样本、错误指标不会生成成功记录。合成分数结果不代表真实模型指标，临时目录已由测试清理。
3. 原生 `verify.py --model /home/zjs/llm/chatglm3-6b` 通过，包含随包资源和 checksums 校验。
4. 沙箱外物理 GPU 0（NVIDIA A100-SXM4-80GB），现有 mia 环境（PyTorch 2.5.1+cu124、Transformers 4.48.3、PEFT 0.20.0）完成开发冒烟：embedding / gradient 各生成 1 member + 1 nonmember 的邻居，并分别加载两侧 adapter 完成真实攻击；example 以 2 条构造记录生成向量、对 1 member + 1 nonmember 完成 mean/private 攻击。共七个原生阶段退出码均为 0，覆盖六个变体。
5. GPU 冒烟记录位于 `/tmp/glm_subj_gpu_smoke_xohymdy6/smoke_manifest.json`，各阶段日志及原生结果在同目录。明确标记 `development_smoke`、`formal_evaluation=false`，没有写入统一工具正式历史；`/tmp` 可能由系统清理。

### 验收边界

目前完成接入、CPU 验证和真实 GPU 小样本运行验证，未运行正式 100/1000、1000/1000 规模实验，也未宣称任何正式攻击指标或隐私保证。外部权重模式的来源校验已做 CPU 测试，尚未对另选外部训练产物运行正式攻击。启动正式实验可使用 README 中的统一 CLI；默认不会缩减样本数或复用开发冒烟结果。
