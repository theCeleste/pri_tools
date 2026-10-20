
## 环境

使用 Linux、Python 3.11 和支持 BF16 的 NVIDIA GPU。本机验证使用 A100 80GB；其他显卡的显存需求未实测。运行时只应暴露一张 GPU。

```bash
conda create -n chatglm-subj-dp python=3.11 -y
conda activate chatglm-subj-dp
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 模型和数据

准备完整的 `THUDM/chatglm3-6b` 模型目录，包括权重、分词器、配置及模型自定义 Python 文件。训练脚本使用 `local_files_only=True`，直接从你传入的本地目录加载模型。

模型配置必须有 `auto_map.AutoModelForSequenceClassification`，指向 `modeling_chatglm.ChatGLMForSequenceClassification`。该类需提供 `classifier_head`；这是本脚本训练二分类头所需的接口。

SUBJ 数据由 `datasets.load_dataset("SetFit/subj")` 自动下载或读取 Hugging Face 本地缓存。首次运行需要能访问数据仓库；离线运行前应自行准备同一数据集的缓存。

## 启动

在此目录执行，将模型路径替换成自己的：

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh /path/to/chatglm3-6b
```

也可直接启动 Python：

```bash
CUDA_VISIBLE_DEVICES=0 python dpsgd_chatglm3_subj.py \
    --model_name /path/to/chatglm3-6b \
    --batch_size 32 \
    --max_physical_batch_size 16 \
    --seq_length 128 \
    --lora_rank 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 3.0 \
    --epochs 4
```

使用 `python dpsgd_chatglm3_subj.py --help` 查看参数。显存不足时可降低 `--max_physical_batch_size`，例如改为 `4`；保持 `--batch_size 32` 可继续使用原逻辑批量大小。若运行 2 轮，显式传入 `--epochs 2`，噪声会按该轮数重新校准。

从项目根目录也可通过统一入口运行：

```bash
python privacy_tool.py run --backend glm_subj --method cliptext --experiment performance_baseline --dry-run
```

将 `python.glm_subj_dpsgd` 配置为上述环境解释器，`glm_subj.base_model` 配置为本地模型目录。
移除 `--dry-run` 执行训练，改用 `--history` 查询结果；`--method` 同样支持 `privsv` 和 `zo_dpga`。
统一入口会通过 `--output_dir` 将本次输出隔离到独立运行目录。

## 配置和输出

- 数据划分：训练 8000；原始 test 2000 条按标签分层，以种子 42 等分为验证 1000 和测试 1000。测试集不参与训练或早停。
- 训练种子：1337。模型为 BF16，优化器 AdamW；LoRA `r=16`、`alpha=32`、`dropout=0.1`，目标模块 `query_key_value`、`dense`，同时训练分类头。
- DP：Opacus Ghost Clipping、Poisson 采样，目标 `epsilon=8`、`delta=5e-6`，裁剪阈值 `3`；会计校准使用实际 `--epochs`，使用默认 PRV 会计。
- 评估：每 400 个物理微批次计算验证准确率（预测标签正确数 / 验证样本数），连续 3 次未提升触发早停。物理微批次数与隐私会计的逻辑更新次数不同，保留原脚本行为。
- 输出写到运行工作目录；`run.sh` 会将工作目录固定到本文件夹。`dp_chatglm3_subj_metrics.csv` 为追加日志，`lora_dp_chatglm3_subj_model_best/` 和 `lora_dp_chatglm3_subj_model_epoch_N/` 保存 LoRA 与分类头参数及分词器。再次运行会复用这些路径；需要保留结果时请先移走旧输出。
- 可用 `--output_dir /path/to/run` 指定输出目录。训练成功后生成 `training_report.json`，包含实际 epsilon、训练参数、checkpoint 路径、训练步骤吞吐量和峰值已分配显存。训练步骤计时使用 CUDA 同步，排除验证和保存；不足 400 个微批次时验证准确率记录为 null。

预算解释以单条文本为保护单位、add/remove 记录级邻接及独立 Poisson 采样为前提，并假设验证集公开。训练损失日志和验证指标未单独做隐私处理；BF16 补丁沿用原实现。本包整理不构成对所有输出或有限精度实现的独立隐私证明。
