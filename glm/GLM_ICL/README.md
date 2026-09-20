# ChatGLM3-6B 示例学习与间接攻击

在SST-2上生成合成示例，分别用原文和合成文本作为prompt，评估成员推断攻击效果。固定设置为 **10 member + 500 non-member，seed=0，不加生成噪声**。

“示例微调”指上下文示例学习，不更新模型权重，也不使用LoRA。评估方法见 [METHOD.md](METHOD.md)。

## 环境与模型

在工作区根目录执行：

```bash
conda activate dp-few-shot
cd GLM_10_500_delivery
bash run.sh --mode check
```

不内置 ChatGLM 原模型，默认读取 `../models/chatglm3-6b`。迁移时需自行准备相同的完整原模型，更换存放位置时：

```bash
export CHATGLM_MODEL_DIR=/你的模型目录/chatglm3-6b
```

模型目录需包含完整权重、索引、tokenizer和自定义Python文件。SST-2数据及BERT权重已包含在交付目录中，无需联网下载。

新机器使用Python 3.10，安装依赖：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

需兼容CUDA 12.8的NVIDIA驱动。保持依赖版本，尤其是Transformers 4.48.3；建议使用单张80 GB GPU。

## 运行

```bash
bash run.sh --mode full --gpu 3       # 生成示例、攻击、融合、评估
```

已有自行生成的示例或攻击结果时，可分步运行：

```bash
bash run.sh --mode attack --gpu 3 --input-json runs/run_01/generated/synthetic_text_results.json
bash run.sh --mode evaluate --results runs/run_01
```

`attack`读取指定的10条示例JSON，重跑攻击和评估；`evaluate`读取指定实验目录的四份攻击CSV，重新融合、计算指标，不使用GPU。输入应来自同一次完整实验。

`--gpu` 指物理卡，可选0/1/2/3，全程单卡。启动前确认该卡可用。GPU模式可加 `--dry-run` 仅查看命令。

结果默认保存到新建的 `runs/` 子目录；可用 `--output runs/实验名称` 指定路径。相对路径以交付目录为基准，已有输出目录不会被覆盖。

后台运行及查看阶段进度：

```bash
nohup bash run.sh --mode full --gpu 3 --output runs/run_01 > progress.log 2>&1 &
tail -f progress.log
```

阶段总览会显示详细日志路径。逐条进度在对应阶段日志中，例如：

```bash
tail -F runs/run_01/generation.log
tail -F runs/run_01/synthetic_text_sst2.log
```

已完成阶段的日志不再更新，退出日志查看不会停止实验。

## 文件与输出

| 目录/文件 | 用途 |
| --- | --- |
| `DPICL/`、`run.sh` | 实验代码与运行入口 |
| `data/` | SST-2数据 |
| `tests/`、`verify.py`、`checksums.json` | 功能测试、依赖和文件完整性校验 |

交付目录不包含历史实验输出。运行后，`runs/实验名称/` 包含：

- `generated/`：生成示例与生成参数。
- `original_text/`、`synthetic_text/`：各自的攻击CSV、融合CSV及 `evaluation/metrics.json`、`roc.csv`、`roc.png`。
- 阶段日志、`run_config.json` 和 `stage_timings.json`：进度、运行参数和耗时。

同一CSV可精确复算指标，更换软硬件后的完整重跑可能有数值差异。

