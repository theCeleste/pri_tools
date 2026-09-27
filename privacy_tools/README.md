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
