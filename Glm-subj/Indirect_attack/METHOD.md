# 实验方法

## 通用攻击

实验数据集为SUBJ，目标模型为ChatGLM3-6B。成员来自相应方法使用的训练或构造数据，非成员来自未参与训练的测试数据；所有固定划分均检查归一化文本无重复、成员与非成员无交集。

BERT base uncased对每个非特殊token的embedding施加 `0.7` dropout，从每个位置保留 `top-k=6` 个替换候选，汇总前50个候选并使用前10个单token替换邻居。固定seed为42。原文和邻居使用同一目标模型与tokenizer，攻击分数为：

```text
score(x) = mean_logprob(x) - mean_i(mean_logprob(neighbor_i))
```

分数越大越倾向成员。成员标签为1，非成员标签为0。阈值从观测分数中选择，在实际FPR不超过目标值时取最高TPR，不进行插值。

## 示例方法

从8000条SUBJ构造记录中按seed 42打乱并选择1000条构造steering向量，这1000条同时作为member。non-member候选池为test前2000条，攻击seed为43；完成攻击后按seed 60固定选择1000条non-member进行评估。Mean和Private共用构造样本、攻击样本、固定子集和邻居。

| 参数 | 值 |
| --- | --- |
| 注入层 | 11、12、13、14、15 |
| 激活读取位置 | `-2` |
| 注入强度 | 256 |
| Private裁剪值 | 18 |
| Private噪声倍率 | 0.005 |
| Private seed | 42 |
| 目标最大长度 | 512 |
| 攻击seed | 43 |
| non-member子集seed | 60 |
| member / non-member | 1000 / 1000 |

Mean Steering使用样本激活差的平均方向；Private Steering先逐样本裁剪，再加入高斯噪声。向量在运行时构造，不更新ChatGLM3-6B权重。

## 嵌入方法

Raw LoRA在原始SUBJ训练文本上训练；ClipText+LoRA在嵌入保护后的训练文本上训练。两者均使用完整7200条训练记录，攻击集合为其中固定100条member及测试集固定1000条non-member。

两套LoRA均采用rank 128、alpha 256、dropout 0，适配 `query_key_value`、`dense`、`dense_h_to_4h`、`dense_4h_to_h`，训练15 epoch。攻击按完整有效文本token计算平均log probability，最大长度512。

## 梯度方法

MeZO为无保护零阶优化，DPZero为加入梯度裁剪和高斯噪声的保护版本。两者均使用rank 1、alpha 2、dropout 0，仅适配第27层 `query_key_value`，训练9000步。固定100条高暴露训练文本作为member，固定1000条测试文本作为non-member。

训练与攻击均以前8个token作为上下文，计算后续token的平均log probability，最大长度14。DPZero采用裁剪阈值0.1、目标 `epsilon=8`、`delta=1e-5`，对应噪声倍率约0.36499。

## 样本分辨率

示例方法的1000条member和1000条non-member使TPR与FPR步长均为0.1%。嵌入和梯度方法的100条member使TPR步长为1%，1000条non-member使FPR步长为0.1%；各方法的0.01% FPR位于零假阳性区域。
