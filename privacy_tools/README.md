# 集成层内部说明

普通使用者只需在项目根目录运行：

```bash
python3 privacy_tool.py
```

主要文件：

- `unified.py`：类别、方法、四项运行菜单和参数问答；
- `catalog.py`：六种方法的训练、对比、直接评估和间接评估命令；
- `bootstrap.py`：环境、依赖、CUDA、模型和数据检查；
- `core.py`：执行、日志与统一指标 CSV；
- `ATTACK_README_ALIGNMENT.md`：六种方法的原始说明、实际攻击脚本与参数对应证明；
- `settings.example.json`：可选的持久配置；
- `requirements/auto/`：自动安装依赖。

每个方法只对外提供四项：

```text
1 本微调方法
2 对比微调方法
3 直接评估
4 间接评估
```

训练结果写入 `privacy_tools/results/<类别>/<方法>/<运行项>/experiment_metrics.csv`，记录隐私参数、Acc、峰值内存、运行时间和吞吐率。直接评估与间接评估按原始 `.md` 流程运行，攻击指标保存在各自单次运行目录中。
