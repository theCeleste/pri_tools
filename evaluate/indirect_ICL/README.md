# 示例学习间接攻击：10 + 500

本包在 **SST-2 / WikiText-2** 上运行 **FM9G-7B / ChatGLM3-6B**，分别以原文和合成文本构造提示，评估间接成员推断攻击。固定使用10条member、500条non-member、seed=0，不加生成噪声。这里的“示例微调”是上下文示例学习，**不更新模型权重，也不使用LoRA**。方法定义见 [METHOD.md](METHOD.md)。

## 准备

两套数据与BERT proposal模型位于交付目录内。请将目标模型放在交付目录同级的 `models/9G7B_MHA` 和 `models/chatglm3-6b`，或通过 `FM9G_MODEL_DIR`、`CHATGLM_MODEL_DIR` 指定模型绝对路径。四种组合使用同一份代码，运行时无需联网下载数据。

建议在Linux、Python 3.10和单张至少80 GB显存的GPU上运行；根据主机驱动安装兼容CUDA 12.8的PyTorch 2.8.0，再安装本包锁定的依赖：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

依赖版本中尤其要保持Transformers 4.48.3。运行前检查文件、依赖、数据路径及所选模型的分词器：

```bash
bash run.sh --mode check --model fm9g
bash run.sh --mode check --model chatglm3
```

模型使用其他位置时，可执行例如 `export FM9G_MODEL_DIR=/模型绝对路径/9G7B_MHA`。`check`不加载完整模型权重；如需逐文件校验模型哈希，可运行 `python verify.py --model fm9g --check-model`，ChatGLM同理。

## 完整实验

在交付目录下执行四条命令；每条命令独立运行，应为同时运行的任务选择不同GPU。`--gpu`是本机物理卡号，可选0、1、2、3。

```bash
bash run.sh --mode full --model fm9g --dataset sst2 --gpu 0
bash run.sh --mode full --model chatglm3 --dataset sst2 --gpu 1
bash run.sh --mode full --model fm9g --dataset wikitext2 --gpu 2
bash run.sh --mode full --model chatglm3 --dataset wikitext2 --gpu 3
```

可用 `--output runs/实验名称` 指定输出目录；默认写入新的 `runs/模型_数据集_full_seed0_时间/`，已有目录不会被覆盖。后台运行及查看进度，例如：

```bash
nohup bash run.sh --mode full --model fm9g --dataset wikitext2 --gpu 3 --output runs/fm9g_wiki > progress.log 2>&1 &
tail -f progress.log
```

运行时依次执行数据准备（仅WikiText-2）、示例生成、原文提示攻击、合成提示攻击、标签融合及指标计算。详细阶段日志和耗时记录在输出目录。两个提示对应的结果分别保存在 `original_text/evaluation/` 和 `synthetic_text/evaluation/`，查看 `metrics.json` 与 `roc.csv`。

已有本包生成的示例时可重跑攻击；已有攻击CSV时可不占GPU、只重算指标。两种模式都要指定原数据集：

```bash
bash run.sh --mode attack --model fm9g --dataset wikitext2 --gpu 3 \
  --input-json runs/fm9g_wiki/generated/synthetic_text_results.json
bash run.sh --mode evaluate --dataset wikitext2 --results runs/fm9g_wiki
```

输出AUC、ROC以及TPR@100%、50%、10%、1%、0.1%、0.01% FPR。

## 文件

| 路径 | 用途 |
| --- | --- |
| `run.sh`、`DPICL/` | 四种组合共用的运行入口、生成、攻击和评估代码 |
| `data/SST-2/`、`data/WikiText-2-raw-v1/` | 固定数据快照 |
| `resources/bert-base-uncased/` | 固定邻居提议模型 |
| `verify.py`、`checksums.json`、`tests/` | 文件完整性及功能检查 |
