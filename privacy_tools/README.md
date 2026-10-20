# 集成层内部说明

普通使用者只需在项目根目录运行：

```bash
python3 privacy_tool.py
```

主要文件：

- `unified.py`：微调/评估任务菜单、方法映射和 JSON 配置接入；
- `catalog.py`：六种方法的训练、对比、直接评估和间接评估命令；
- `bootstrap.py`：环境、依赖、CUDA、模型和数据检查；
- `core.py`：执行、日志与统一指标 CSV；
- `ATTACK_README_ALIGNMENT.md`：六种方法的原始说明、实际攻击脚本与参数对应证明；
- `settings.example.json`：可选的持久配置；
- `requirements/auto/`：自动安装依赖。

交互入口按任务组织：

```text
微调 → 扰动类别 → 具体方法 → 配置模型/数据集 → 当前方法 或 对比方法
评估 → 直接/间接 → 显式/隐式 → 微调类别 → 具体方法 → 配置模型
```

模型、数据集、Python 解释器和全部实验参数均从 `settings.example.json` 或用户指定的 JSON 配置读取。模型和数据集菜单只用于确认配置值；ChatGLM / FM9G 的无扰动间接评估已接入 evaluate/ 新后端，直接评估仍预留。

训练结果写入 `privacy_tools/results/<类别>/<方法>/<运行项>/experiment_metrics.csv`，记录隐私参数、Acc、峰值内存、运行时间和吞吐率。直接评估与间接评估按原始 `.md` 流程运行，攻击指标保存在各自单次运行目录中。

真实运行失败或被中断时只保留诊断日志和 manifest，终端提示“运行失败，请重新运行”；此时不加载历史结果，也不写入实验指标 CSV。

- indirect_evaluation.py：ChatGLM / FM9G 显式、隐式间接评估的共享执行层。
- indirect_data_check.py：成员来源、数据划分与后端运行环境检查。
- [新版间接评估接入说明](INDIRECT_EVALUATION.md)：配置、历史查询、命令预览和离线重算。

### ChatGLM3-6B / Subj（Glm-subj 后端）

ClipText、PrivSV、ZO-DPGA 的微调和评估菜单现可选择 ChatGLM3-6B。
微调仅新增支持 ChatGLM3-6B + Subj；评估选择该模型后自动使用 Subj。
该分支提供立即执行、查看历史、命令预览三种模式，使用独立的 `glm_subj`
配置和 `python.glm_subj` 环境（依赖见 `Glm-subj/requirements.txt`）。

```bash
# 训练命令预览；baseline 是同一方法的非保护效用对照
python privacy_tool.py run --backend glm_subj --method cliptext --experiment utility --dry-run
python privacy_tool.py run --backend glm_subj --method privsv --experiment baseline --dry-run
# 读取匹配身份的最近成功结果
python privacy_tool.py run --backend glm_subj --method cliptext --experiment utility --history
# 已完成对应两侧训练后，预览成对直接攻击
python privacy_tool.py run --backend glm_subj --method zo_dpga --experiment direct_attack --dry-run
# ChatGLM 性能对比：调用 dpsgd/chatglm（同样支持 privsv、zo_dpga）
python privacy_tool.py run --backend glm_subj --method cliptext --experiment performance_baseline --dry-run
python privacy_tool.py run --backend glm_subj --method cliptext --experiment performance_baseline --history
```

`performance_baseline` 使用 `python.glm_subj_dpsgd` 解释器和
`dpsgd/chatglm/requirements.txt`，环境安装步骤见该目录 README。
通过 `glm_subj.dpsgd` 配置批量大小、物理微批量、序列长度、LoRA rank、
学习率、裁剪阈值和训练轮数；默认 epsilon=8、delta=5e-6、种子 1337。
`common.gpu` 必须指定单张 GPU。基础模型沿用 `glm_subj.base_model`，
需包含 `AutoModelForSequenceClassification` 映射和对应分类模型源码。
该脚本从 Hugging Face 缓存或网络加载 `SetFit/subj`，使用其原始 train，
将原始 test 分层拆为 validation/test 各 1000 条，不读取 `glm_subj.raw_data`。
输出独立保存到 `glm_subj.output_root` 下的本次运行目录，含 checkpoint、
CSV、日志、manifest 和 `training_report.json`，可按原方法查询成功历史。
报告记录训练步骤耗时、吞吐量、峰值已分配显存和实际 epsilon；
吞吐量仅计算同步后的训练步骤，不包含加载、验证和保存耗时。
分类头 checkpoint 用于性能对比，不会作为直接攻击的生成式 adapter 自动解析。

移除 `--dry-run` 会实际执行。运行前配置 `glm_subj.base_model`、`raw_data`、
`embedding_data` 和 `output_root`；外部已有模型可填写 `glm_subj.artifacts` 中
各方法的保护/基线 adapter 和 steering 路径。未填写时从同方法的成功训练
manifest 解析产物。PrivSV 的 `raw_adapter` 优先指定两侧共同使用的 Raw LoRA；
未指定时复用已有 Raw LoRA，首次运行则先训练 Raw LoRA。Steering 构造沿用
上游在 base model 上提取向量的实现，推理时将向量注入 Raw LoRA。
Private Steering 当前没有 epsilon accountant，不能宣称形式化 DP 保证。

三种方法均使用原始 Subj `test` holdout、固定 subjective/objective 提示，
按两个标签续写的平均 token log likelihood 选择类别，记录 accuracy、正确数、
分母及 holdout SHA256。`utility_samples=0` 表示全部 holdout；调整该值后应对
保护和基线两侧重新评估。PrivSV 的 Mean/Private 也使用同一评估器。
训练报告中的 loss/perplexity 另行保存，不能视作分类准确率。

直接评估只调用 known-prefix causal extraction，两侧使用相同的原始 Subj
source reference、固定 raw prompt 模式、seed 和攻击参数，并由 `compare.py`
验证目标和配置匹配。保存 train、holdout、差值、命中数/分母及 comparison。
本阶段 `performance_baseline`（DP-SGD）和 `indirect_attack` 仅显示“尚未接入”，
不执行实验或生成成功记录。

实际运行写入 `Glm-subj/outputs/<run_id>/run_manifest.json`、`models/`、
`attacks/` 和 `run.log`。只有所有步骤成功且报告齐全才标记成功；历史按
后端、方法、模型、数据集、实验、角色和成功状态过滤，再按 manifest 完成时间
取最新结果。旧目录没有 manifest 的实验不会自动混入本分支历史。
预览和历史读取均不执行命令、不创建运行记录。
