# SUBJ 间接成员推断评估

本包使用ChatGLM3-6B评估三种微调或模型适配方法的间接成员推断风险：示例方法、嵌入方法和梯度方法。包内提供固定数据、方法权重、BERT邻居提议模型和评估代码；ChatGLM3-6B底座由运行者指定。方法和参数见 [METHOD.md](METHOD.md)。

## 准备

建议使用Linux、Python 3.10、单张CUDA GPU和PyTorch 2.5.1。安装依赖：

```bash
python -m pip install -r requirements.txt
```

底座模型默认位于交付包同级的 `models/chatglm3-6b`。也可以通过环境变量或命令行指定：

```bash
export CHATGLM_MODEL_DIR=/path/to/chatglm3-6b
./run.sh --mode check
```

## 完整运行

三条命令分别生成邻居、执行保护前后攻击并计算指标。`--gpu`填写物理GPU编号。

```bash
./run.sh --mode full --method example   --gpu 0
./run.sh --mode full --method embedding --gpu 1
./run.sh --mode full --method gradient  --gpu 2
```

默认输出到 `runs/` 下的新目录。可固定输出位置：

```bash
./run.sh --mode full --method embedding --gpu 1 --output runs/embedding
```

结果位于各方法输出目录的 `evaluation/metrics.json` 和 `metrics.csv`：

- 示例方法：`mean/`、`private/`
- 嵌入方法：`raw/`、`cliptext_lora/`
- 梯度方法：`mezo/`、`dpzero/`

示例方法最终评估固定的1000条member和1000条non-member。运行时先按固定顺序完成1000+2000条攻击，再按固定seed选择1000条non-member，以复现实验划分。

## 分阶段运行

仅攻击时，可选择单个变体并复用已有邻居文件：

```bash
./run.sh --mode attack --method embedding --variant raw --gpu 1 \
  --neighbors runs/embedding/neighbors.jsonl --output runs/embedding_raw
```

仅重新计算指标时不加载GPU模型：

```bash
./run.sh --mode evaluate --method embedding --output runs/embedding
```

查看命令但不执行：

```bash
./run.sh --mode full --method gradient --gpu 2 --dry-run
```

后台运行和查看进度：

```bash
nohup ./run.sh --mode full --method gradient --gpu 2 --output runs/gradient > gradient.progress 2>&1 &
tail -f gradient.progress
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `example/` | Mean Steering与Private Steering构造及攻击 |
| `embedding/` | Raw LoRA与ClipText+LoRA权重及攻击 |
| `gradient/` | MeZO与DPZero权重及攻击 |
| `resources/` | BERT邻居提议模型 |
| `run.sh`、`run.py` | 统一运行入口 |
| `verify.py`、`tests/` | 完整性检查和基础测试 |

评估输出ROC AUC以及TPR@100%、50%、10%、1%、0.1%、0.01% FPR。

## 统一工具接入与资源覆盖

项目统一入口已接入本包，参见 [集成说明](../../privacy_tools/README.md#chatglm3-6b--subj-间接评估)。
`run.py` 额外接受 `--proposal-model`、`--baseline-adapter`、`--protected-adapter`、
`--splits`、`--interface-dir`；省略时保持包内默认资源。baseline/protected 分别对应
embedding 的 raw/cliptext_lora、gradient 的 mezo/dpzero。
外部 adapter 必须使用与训练来源匹配的攻击划分；统一工具负责来源记录校验。
example 的 interface-dir 同时决定构造向量和攻击样本的数据，不加载 LoRA。
`full` 总是运行两个变体；embedding/gradient 复用邻居请使用 attack 模式。
