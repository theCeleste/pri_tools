# 间接攻击评估方法

## 流程与数据

对每个模型和数据集，先用底座模型生成10条合成示例，再分别用原文和合成文本的后5条（`data[5:10]`）构造提示。两种提示下各攻击10条member和500条non-member，分别融合标签、计算指标。模型权重不更新，不添加生成噪声。

- **SST-2**：`train.tsv`为私有示例采样池；`dev.tsv`用于20条情感分类效用测试；`test.tsv`中过滤长度不超过10的文本，依序取前500条作为non-member。生成提示包含情感标签。
- **WikiText-2**：使用 `wikitext-2-raw-v1` 数据，按train、validation、test顺序过滤 `len(text.strip()) <= 10` 的文本，并对空白规范化后的文本去重；test中前500条作为non-member，train作为私有采样池。它没有情感类别，生成和攻击提示均不包含情感标签，也不做分类效用测试。

生成时每个token从私有池抽取20条文本，分成10组、每组2条，聚合目标模型概率后贪心生成；最多生成约100个token，遇到停止符或EOS提前结束。10条生成记录的 `original_text` 作为member；500条test文本作为non-member。融合标签为member=1、non-member=0，和SST-2的情感标签无关。

## 攻击与指标

BERT base uncased产生单token替换邻居：被替换位置的embedding dropout为0.7，每个位置取top-k=6，按候选分数保留最多50条，实际使用前10条评分。原文和邻居使用相同的目标模型tokenizer及有效正文token平均分数；评分排除提示和padding。SST-2的目标模型及BERT最大长度均为512；WikiText-2的目标模型最大长度2048、BERT最大长度512。

```text
score(x) = mean_logprob(x | prompt)
           - mean_i(mean_logprob(neighbor_i | prompt))
```

分数越大越像member；它是论文中 `loss(x)-mean(loss(neighbor))` 的反号。两种提示分别计算 AUC、ROC 及 TPR@1%、0.1%、0.01% FPR。固定 FPR 指标取 ROC 上实际 FPR 不超过目标值时的最大 TPR，不插值。TPR 以百分比形式显示，例如 `TPR@1%FPR: 40.00%`。

| 固定参数 | 值 |
| --- | --- |
| 目标模型 | FM9G-7B或ChatGLM3-6B原始底座 |
| seed / 生成精度 / 攻击精度 | 0 / FP16 / FP32 |
| member / non-member / 提示示例 | 10 / 500 / 后5条 |
| BERT dropout / top-k / 评分邻居 | 0.7 / 6 / 前10条 |
| 评分模式 | `token-mean` |

10条member使TPR最小步长为10%；500条non-member使FPR最小非零步长为0.2%，因此0.1%和0.01%均在零假阳性区域求TPR，但结果不必为0。

只有后5条原文直接进入攻击提示，全部10条原文均标为member；生成时使用的私有样本另行抽取。这里的member不表示参与过权重训练，且无噪声设置不提供差分隐私保证。
