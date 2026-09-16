import pandas as pd

# 1. 读取 label_flipping_results.json 数据
import json
json_path = "synthetic_text_results.json"
# json_path = "./sst2_train.json"

# 读取JSON文件
with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 将JSON数据转换为DataFrame
data_list = []
for item in json_data:
    data_list.append({
        "sentence": item["original_text"],
        "label": item["label"]
    })

sst2 = pd.DataFrame(data_list)
sst2_sentences = set(sst2["sentence"].astype(str).str.strip())

# 2. 读取前缀攻击提取的文本
# attack_path = "outputs/custom_prefix_samples.txt"
attack_path = "outputs/custom_prefix_samples_non.txt"
with open(attack_path, "r", encoding="utf-8", errors="ignore") as f:
    attack_text = f.read()

# 按行或句号拆分
import re
attack_lines = [s.strip() for s in re.split(r'[:\n\.!?]', attack_text) if s.strip()]
attack_lines_set = set(attack_lines)

# 3. 统计重合句子
intersection = attack_lines_set & sst2_sentences

# print(f"共有 {len(intersection)} 条句子在 隐私 训练集中出现。")
# print(f"占前缀攻击文本的比例为 {len(intersection) / len(attack_lines_set):.2%}")
# print("\n示例匹配句子（前10条）:")
# for s in list(intersection)[:10]:
#     print("-", s)
