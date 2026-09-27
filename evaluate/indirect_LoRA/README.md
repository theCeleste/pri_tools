# LoRA 间接成员推断评估

本包对 FM9G-7B、ChatGLM3-6B 的 SST-2 / WikiText-2 LoRA 模型进行邻居文本成员推断。每组使用固定划分中的前 1,200 条成员和前 1,200 条非成员，输出 ROC AUC 及 TPR@100%、50%、10%、1%、0.1%、0.01% FPR。

## 准备

需要 Python 3.10、CUDA 可用的 PyTorch 2.5.1，以及 `requirements.txt` 中的依赖。先在当前环境安装所列依赖：

```bash
python -m pip install -r requirements.txt
./run.sh --mode check
```

可用 `python -c 'import torch; print(torch.cuda.is_available())'` 确认当前环境能访问 GPU。

底座模型路径默认为交付包上一级的 `models/9G7B_MHA` 和 `models/chatglm3-6b`。若放在其他位置，可分别设置 `FM9G_MODEL_DIR`、`CHATGLM_MODEL_DIR` 环境变量。检查命令会验证文件校验和、依赖、数据和四组模型文件。

## 运行四组实验

在交付包目录执行，`--gpu` 填物理 GPU 编号 0、1、2 或 3；每个命令分别运行成员攻击、非成员攻击、评估三个阶段：

```bash
./run.sh --model fm9g --dataset sst2 --gpu 0
./run.sh --model fm9g --dataset wikitext2 --gpu 1
./run.sh --model chatglm3 --dataset sst2 --gpu 2
./run.sh --model chatglm3 --dataset wikitext2 --gpu 3
```

可逐条串行运行，也可在四张 GPU 上同时运行。默认输出至 `runs/` 下的独立目录；用 `--output /path/to/new-run` 指定位置。终端逐阶段显示 `START/DONE`，详细进度在对应的 `attack_member.log`、`attack_nonmember.log` 中。结果见 `mia_evaluation_results.csv`；`stage_timings.json` 记录阶段耗时。

查看命令而不执行：

```bash
./run.sh --model fm9g --dataset sst2 --gpu 0 --dry-run
```

仅用已生成的分数重新评估：

```bash
./run.sh --mode evaluate --model fm9g --dataset sst2 \
  --scores-dir /path/to/scores --output /path/to/new-evaluation
```

`--scores-dir` 中须有 `member_scores.pkl` 和 `nonmember_scores.pkl`，每个文件各含 1,200 条有效评分。评估只读这些文件，不加载 GPU 模型。四组实验的方法与参数见 [METHOD.md](METHOD.md)。

## 统一入口的外部输入

run.py 支持 --adapter-dir、--tokenizer-dir、--neighbor-model-path、--dataset-dir，
或成对的 --member-data-file / --nonmember-data-file。文件输入与 --dataset-dir 互斥。
这些参数仅用于 attack 模式；不传时保持交付包默认资源与划分。
相对路径仍相对于此包，统一入口传入绝对路径。
--dry-run 只构造命令、不校验资源存在性，真实运行仍检查资源。
项目统一入口在启动攻击前额外验证成员来源，详见 ../../privacy_tools/INDIRECT_EVALUATION.md。
