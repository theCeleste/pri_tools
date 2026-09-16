import pandas as pd
import re
import csv
import random

# 1) 读取 SST-2 训练集 TSV
tsv_path = "../../privacy_tools/datasets/sst2/train.tsv"
sst2 = pd.read_csv(
    tsv_path,
    sep="\t",
    header=0,                 # 第一行是：sentence\tlabel
    dtype={"sentence": str},  # sentence 强制按字符串读
    quoting=csv.QUOTE_NONE,   # 避免引号相关解析问题（一般 SST-2 不需要，但更稳）
    keep_default_na=False     # 避免空字符串变成 NaN
)

# 句子集合（去首尾空白）
sst2_sentences = set(sst2["sentence"].astype(str).str.strip())

# （可选）做一个 sentence -> label 的映射，后面展示匹配样例时能把标签也打印出来
sent2label = dict(zip(sst2["sentence"].astype(str).str.strip(), sst2["label"]))

# 2) 读取前缀攻击提取的文本
attack_path = "./outputs/top_n_samples.txt"
# attack_path = "./outputs/top_n_samples-sv.txt"
with open(attack_path, "r", encoding="utf-8", errors="ignore") as f:
    attack_text = f.read()

# 按行或标点拆分（保留你原来的规则）
attack_lines = [s.strip() for s in re.split(r'[:\n\.!?]', attack_text) if s.strip()]
attack_lines_set = set(attack_lines)

# 3) 统计重合句子
intersection = attack_lines_set & sst2_sentences

denom = len(attack_lines_set)
ratio = (len(intersection) / denom) if denom else 0.0
# print(attack_lines_set)
# print(intersection)

# print(f"共有 {len(intersection)} 条句子在 SST-2 训练集中出现。")
# print(f"占前缀攻击文本的比例为 {ratio:.2%}")

# print("\n示例匹配句子（前5条）:")
# for s in list(sst2_sentences)[:len(intersection)][:5]:
#     # 如果你不想显示 label，把下面这一行改成 print("-", s) 即可
#     # print(f"- {s} \t(label={sent2label.get(s)})")
#     print("-", s)
    
