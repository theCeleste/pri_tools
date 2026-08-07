from datasets import load_dataset
import json
from tqdm import tqdm

# 1. 加载 SST-2
dataset = load_dataset("stanfordnlp/sst2", split="train[1500:2000]")

def sst2_to_psa_format(example):
    sentence = example["sentence"]
    label = example["label"]

    question = (
        "What is the sentiment of the following sentence?\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        "Options:\n"
        "(A) The sentiment is positive.\n"
        "(B) The sentiment is negative."
    )

    if label == 1:  # positive
        answer_matching_behavior = "(A) The sentiment is positive."
        answer_not_matching_behavior = "(B) The sentiment is negative."
    else:  # negative
        answer_matching_behavior = "(B) The sentiment is negative."
        answer_not_matching_behavior = "(A) The sentiment is positive."

    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior
    }

# 2. 转换整个数据集
psa_style_data = []
for ex in tqdm(dataset):
    psa_style_data.append(sst2_to_psa_format(ex))

# 3. 保存为 JSON
output_path = "datasets/generate/sst2/generate_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(psa_style_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(psa_style_data)} examples to {output_path}")
