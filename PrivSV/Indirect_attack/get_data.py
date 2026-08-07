import json
from datasets import load_dataset

# 1. 加载 SST-2 数据集
dataset = load_dataset("stanfordnlp/sst2", split="train[1500:2000]")

# 2. 标签映射
label_map = {
    0: "Negative",
    1: "Positive"
}

# 3. 转换格式
output_data = []
for idx, example in enumerate(dataset):
    item = {
        "index": idx,
        "text": example["sentence"],
        "label": label_map[example["label"]]
    }
    output_data.append(item)

# 4. 保存为 JSON 文件
with open("sst2_train1.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("保存完成，共样本数：", len(output_data))
