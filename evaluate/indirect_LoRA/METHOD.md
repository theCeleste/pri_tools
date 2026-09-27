# 实验方法

## 数据与模型

`data/sst2/` 和 `data/wikitext2/` 分别提供固定的 `train`、`nonmember` 划分。文本按首尾空白、连续空白和大小写归一化后去重；`nonmember` 与完整 `train` 无重叠。成员取 `train` 前 1,200 条，非成员取 `nonmember` 前 1,200 条。成员属于 LoRA 微调数据，非成员未参与微调。

四组已保存的适配器位于 `weights/{model}/{dataset}/`。FM9G 两组使用 3 epoch、r=64、alpha=32、dropout=0、仅 `q_proj/k_proj/v_proj` 的 LoRA；SST-2 训练文本为评论正文，WikiText-2 为原文。ChatGLM3 两组使用 r=16、alpha=32、dropout=0，适配 `query_key_value/dense/dense_h_to_4h/dense_4h_to_h`；SST-2 训练文本包含 `Review`、`Sentiment` 和情绪标签，WikiText-2 为原文。评估加载底座模型和对应适配器，不重新训练。

## 邻居攻击

攻击对每条文本的正文及 EOS 评分，不把数据标签送入攻击。RoBERTa-base 遮蔽各非特殊 token，取每个位置概率最高的 8 个替换候选；一个邻居只替换一个 token，解码去重后最多保留 100 个邻居。该实现采用掩码语言模型提议，不使用 embedding dropout。

原文和邻居使用同一目标模型、同一目标 tokenizer 和 256 token 截断。每条文本的有效预测 token 交叉熵先取平均，再取负值作为平均 log probability；邻居评分 batch size 为 8。固定 seed 为 42，目标精度 BF16。

```text
score(x) = mean_logprob(x) - mean_i(mean_logprob(neighbor_i))
```

分数越大越倾向成员；仅当 `score > threshold` 时判为成员。该分数与以 loss 表示的邻居差值方向相反，但排序判定等价。

## 指标

成员标签为 1，非成员标签为 0。使用 1,200 + 1,200 条攻击分数计算 ROC AUC，并在目标 FPR 为 100%、50%、10%、1%、0.1%、0.01% 时计算 TPR。阈值取自观测分数及高于最大分数的候选，选择实际 FPR 不超过目标且 TPR 最大者；不做插值。

TPR = 正确识别的成员数 / 1,200；FPR = 错判为成员的非成员数 / 1,200。因此非零 FPR 的最小步长是 1/1,200，约 0.0833%；0.01% 目标对应零误报。输出同时包含实际 FPR、阈值和混淆矩阵计数。
