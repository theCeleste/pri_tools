# ChatGLM3-6B SST-2 LoRA 与间接攻击

包含 LoRA 权重、固定数据划分及完整训练与攻击代码，不内置 ChatGLM 原模型。训练使用 33,579 条文本，攻击评估使用 1,200 member + 1,200 non-member；seed=42。评估方法见 [METHOD.md](METHOD.md)。

## 环境

在工作区根目录执行：

```bash
conda activate mia
cd GLM_LoRA_1200_1200_delivery
bash run.sh --mode check
```

默认读取 `../models/chatglm3-6b`。迁移到其他机器时，需自行准备相同的完整原模型；可指定其位置：

```bash
export CHATGLM_MODEL_DIR=/你的模型目录/chatglm3-6b
```

模型目录需包含完整权重、索引、tokenizer 和自定义 Python 文件；LoRA 适配器不能独立运行。新机器建议 Python 3.10，安装：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip check
```

需支持 CUDA 12.4、BF16 的 NVIDIA GPU，建议单张 80 GB 卡。SST-2 数据及 RoBERTa 邻居模型已包含，无需下载；使用外置原模型须遵守其许可证。

## 运行

直接使用随包 LoRA 权重，重新攻击和评估：

```bash
bash run.sh --mode attack --gpu 3 --output runs/attack_01
```

重新微调，然后自动攻击、评估：

```bash
bash run.sh --mode full --gpu 3 --output runs/full_01
```

可选分步运行：

```bash
bash run.sh --mode train --gpu 3 --output runs/train_01
bash run.sh --mode attack --gpu 3 --adapter-dir runs/train_01/adapter --output runs/attack_02
bash run.sh --mode evaluate --scores-dir runs/attack_01 --output runs/evaluate_01
```

`--gpu` 可选物理卡 0/1/2/3，全程只使用所选单卡；先确认显存可用。评估模式不使用 GPU。相对路径以交付目录为基准，已有输出目录不会覆盖；加 `--dry-run` 可只查看命令。

后台执行与查看进度：

```bash
nohup bash run.sh --mode full --gpu 3 --output runs/full_01 > progress.log 2>&1 &
tail -f progress.log
tail -F runs/full_01/train.log
tail -F runs/full_01/attack_train.log
tail -F runs/full_01/attack_nonmember.log
```

阶段总览显示各阶段日志路径，已完成阶段的日志不再更新。退出日志查看不会停止实验。

## 文件与输出

| 目录/文件 | 用途 |
| --- | --- |
| `weights/lora/` | LoRA 适配器及对应 tokenizer，原模型外置 |
| `data/` | 固定顺序的 train、nonmember、validation 划分 |
|                                         |                                              |
| `code/`、`run.sh`、`run.py` | 微调、攻击、评估及运行入口 |
| `tests/`、`verify.py`、`checksums.json` | 功能测试、依赖与文件完整性检查 |

输出目录包含逐阶段日志、成员与非成员的 `*_scores.pkl`、样本元信息、`mia_evaluation_results.csv`、运行参数和阶段耗时。微调模式另生成 `adapter/`；随包权重不会被覆盖。

指标包括 AUC、TPR@100%、50%、10%、1%、0.1%、0.01% FPR，以及实际 FPR、阈值和混淆矩阵计数。交付目录不附带历史日志或评估结果；同一分数文件可精确复算指标，重新训练或更换软硬件后不保证数值逐位相同。
