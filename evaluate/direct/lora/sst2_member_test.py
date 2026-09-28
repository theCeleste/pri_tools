import os
import re
import csv
import json
import html
import difflib
from datetime import datetime
from collections import Counter, defaultdict

from datasets import load_from_disk


# ============================================================
# 路径配置
# ============================================================

train_dataset_path = "/home/wym/projects/ExtractingModel/Extractingcode/finetune/data/sst2/train"
# 修改为您的文件路径（支持JSONL或TXT）
attack_path = "/home/wym/projects/ExtractingModel/Extractingcode/save/results/llama_sst2/top100_by_ppl_low_to_high_20260714_134512.txt"

save_dir = "/home/wym/projects/ExtractingModel/Extractingcode/save/results/llama_sst2/ppl"
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 参数
# ============================================================

FUZZY_THRESHOLD = 0.85       # 模糊匹配阈值
CONTAINS_MIN_LEN = 50        # contains 匹配最短长度
TOP_K_CANDIDATES = 1000      # 每条生成文本最多和多少条训练样本做 fuzzy 精排

LABEL_NAMES = {
    0: "Negative",
    1: "Positive"
}


# ============================================================
# 文本处理函数
# ============================================================

def normalize_basic(text):
    """
    基础归一化：
    - 小写
    - HTML 实体转换
    - 空白归一化
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("#39;", "'")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = html.unescape(text)

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_loose(text):
    """
    更宽松的归一化：
    - 在 basic 基础上去掉大部分标点
    - 用于 fuzzy / contains
    """
    text = normalize_basic(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def similarity(a, b):
    """
    difflib 相似度。
    """
    a = normalize_loose(a)
    b = normalize_loose(b)

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def best_similarity(candidate, gold):
    """
    兼顾生成文本过长、过短的情况。
    """
    cand = normalize_loose(candidate)
    ref = normalize_loose(gold)

    if not cand or not ref:
        return 0.0

    scores = []

    # 直接比
    scores.append(difflib.SequenceMatcher(None, cand, ref).ratio())

    # 如果生成文本更长，只取和 gold 等长附近的前半段
    if len(cand) >= len(ref):
        scores.append(difflib.SequenceMatcher(None, cand[:len(ref)], ref).ratio())

    # 如果生成文本更短，只和 gold 前半段比
    if len(ref) >= len(cand):
        scores.append(difflib.SequenceMatcher(None, cand, ref[:len(cand)]).ratio())

    return max(scores)


def is_valid_candidate(text):
    """
    过滤明显无效的生成候选。
    """
    if not text:
        return False

    text = text.strip()

    if len(text) < 20:
        return False

    letter_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if letter_ratio < 0.15:
        return False

    if not re.search(r"[a-zA-Z]{3,}", text):
        return False

    unique_chars = len(set(text))
    if unique_chars < 8:
        return False

    return True


def clean_text_for_matching(text):
    """
    清理文本用于匹配：
    - 移除 [gMASK]sop 等特殊标记
    - 移除多余的空白
    """
    if not text:
        return text
    
    # 移除 [gMASK]sop, [gMASK], [MASK] 等标记
    text = re.sub(r'\[gMASK\]sop\s*', '', text)
    text = re.sub(r'\[[A-Z_]+\]\s*', '', text)
    
    # 移除 @-@, @.@ 等特殊标记
    text = re.sub(r'@-@', '', text)
    text = re.sub(r'@\.@', '', text)
    text = re.sub(r'@,@', '', text)
    
    # 移除多余的HTML标签残留
    text = re.sub(r'<[/]?[a-z]+>', '', text)
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# ============================================================
# 从文件加载候选样本（支持JSONL和TXT）
# ============================================================

def load_candidates_from_file(file_path):
    """
    从文件加载候选样本，自动检测格式：
    1. JSONL: 每行一个JSON对象
    2. TXT: 格式化的文本文件，包含 [Rank] 和 Text: 字段
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件：{file_path}")
        return [], []
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    print(f"检测到文件格式: {ext}")
    
    if ext in ['.json', '.jsonl']:
        return load_candidates_from_jsonl(file_path)
    elif ext == '.txt':
        return load_candidates_from_txt(file_path)
    else:
        print(f"未知文件格式 {ext}，尝试自动检测...")
        try:
            return load_candidates_from_jsonl(file_path)
        except:
            try:
                return load_candidates_from_txt(file_path)
            except:
                print("无法解析文件格式")
                return [], []


def load_candidates_from_jsonl(jsonl_path):
    """
    从JSONL文件加载候选样本
    """
    candidates = []
    candidates_metadata = []
    
    print(f"从JSONL文件加载候选样本: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                candidate = extract_candidate_from_json(data, line_num)
                if candidate:
                    candidates.append(candidate['text'])
                    candidates_metadata.append(candidate['metadata'])
                    
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行JSON解析失败: {e}")
                continue
    
    print(f"从JSONL文件加载了 {len(candidates)} 个候选样本")
    return candidates, candidates_metadata


def extract_candidate_from_json(data, idx):
    """
    从JSON对象中提取候选文本
    """
    text = None
    
    # 优先使用 'text' 字段
    if 'text' in data:
        text = data['text']
    # 其次使用 'generated' 字段
    elif 'generated' in data:
        text = data['generated']
    # 再次使用 'sample' 字段
    elif 'sample' in data:
        text = data['sample']
    # 如果都没有，使用整个JSON的字符串表示
    else:
        text = json.dumps(data, ensure_ascii=False)
    
    if not text:
        return None
    
    # 清理文本
    cleaned_text = clean_text_for_matching(text)
    
    if not is_valid_candidate(cleaned_text) and not is_valid_candidate(text):
        return None
    
    metadata = {
        'id': data.get('id', idx),
        'prompt': data.get('prompt', ''),
        'ppl': data.get('ppl', 0),
        'zlib': data.get('zlib', 0),
        'ratio': data.get('ratio', 0),
        'ppl_zlib_ratio': data.get('ppl_zlib_ratio', 0),
        'full_data': data
    }
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'metadata': metadata
    }


def load_candidates_from_txt(txt_path):
    """
    从TXT文件加载候选样本
    格式示例：
    ================================================================================
    Dataset: sst2
    Top 100 samples by PPL/ZLIB ratio (lower is better)
    ================================================================================
    
    [1] Rank: 1
    Prompt: 
    PPL: 1.9365
    ZLIB: 233
    Ratio: 0.121242
    Text: [gMASK]sop 2004 Asia Pacific...
    --------------------------------------------------------------------------------
    """
    candidates = []
    candidates_metadata = []
    
    print(f"从TXT文件加载候选样本: {txt_path}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 跳过开头的标题部分（从第一个 [数字] 开始）
    first_rank = re.search(r'\[\s*\d+\s*\]', content)
    if first_rank:
        content = content[first_rank.start():]
    
    # 按分隔符分割（每个样本用 --- 或 === 分隔）
    separator_pattern = r'\n\s*[\-=]{10,}\s*\n'
    parts = re.split(separator_pattern, content)
    parts = [p.strip() for p in parts if p.strip()]
    
    print(f"按分隔符分割得到 {len(parts)} 个部分")
    
    # 如果分割结果太少，尝试按 [数字] 分割
    if len(parts) <= 1:
        print("按分隔符分割失败，尝试按 [数字] 分割...")
        pattern = r'(\[\d+\].*?)(?=\n\[\d+\]|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            parts = [m.strip() for m in matches]
            print(f"按 [数字] 分割得到 {len(parts)} 个部分")
    
    # 解析每个部分
    valid_count = 0
    for i, part in enumerate(parts):
        result = parse_txt_sample(part)
        if result:
            candidates.append(result['text'])
            candidates_metadata.append(result['metadata'])
            valid_count += 1
        else:
            # 打印调试信息
            if i < 5:  # 只打印前5个失败的部分
                print(f"警告: 第 {i+1} 个部分解析失败，前100字符: {part[:100]}...")
    
    print(f"成功解析 {valid_count} 个样本")
    return candidates, candidates_metadata


def parse_txt_sample(part):
    """
    解析单个TXT样本块
    """
    if not part:
        return None
    
    # 提取字段
    prompt_match = re.search(r'Prompt:\s*(.+?)(?:\n|$)', part, re.IGNORECASE)
    ppl_match = re.search(r'PPL:\s*([\d.]+)', part, re.IGNORECASE)
    zlib_match = re.search(r'ZLIB:\s*(\d+)', part, re.IGNORECASE)
    ratio_match = re.search(r'Ratio:\s*([\d.]+)', part, re.IGNORECASE)
    
    # 提取Text - 从 "Text:" 开始到下一个分隔符或文件结束
    text_match = re.search(r'Text:\s*(.+?)(?=\n\s*[\-=]{10,}|\Z)', part, re.DOTALL | re.IGNORECASE)
    
    if not text_match:
        # 尝试直接提取 "Text:" 后面的内容
        text_start = re.search(r'Text:', part, re.IGNORECASE)
        if text_start:
            text_content = part[text_start.end():].strip()
            # 截断到下一个分隔符
            next_sep = re.search(r'\n\s*[\-=]{10,}', text_content)
            if next_sep:
                text_content = text_content[:next_sep.start()]
            text_content = text_content.strip()
        else:
            return None
    else:
        text_content = text_match.group(1).strip()
    
    if not text_content:
        return None
    
    # 清理文本
    cleaned_text = clean_text_for_matching(text_content)
    
    if not is_valid_candidate(cleaned_text) and not is_valid_candidate(text_content):
        return None
    
    # 提取rank
    rank_match = re.search(r'\[\s*(\d+)\s*\]', part)
    rank = int(rank_match.group(1)) if rank_match else 0
    
    # 提取元数据
    prompt = prompt_match.group(1).strip() if prompt_match else ''
    ppl = float(ppl_match.group(1)) if ppl_match else 0.0
    zlib = int(zlib_match.group(1)) if zlib_match else 0
    ratio = float(ratio_match.group(1)) if ratio_match else 0.0
    
    metadata = {
        'id': rank,
        'prompt': prompt,
        'ppl': ppl,
        'zlib': zlib,
        'ratio': ratio,
        'ppl_zlib_ratio': ratio
    }
    
    return {
        'text': text_content,
        'cleaned_text': cleaned_text,
        'metadata': metadata
    }


# ============================================================
# 加载训练集
# ============================================================

print("=" * 60)
print("Loading SST-2 train dataset")
print("=" * 60)

dataset = load_from_disk(train_dataset_path)

train_records = []
raw_train_map = {}
basic_train_map = {}
loose_train_map = {}

dataset_label_count = defaultdict(int)

for idx, item in enumerate(dataset):
    text = item["sentence"].strip()
    label = int(item["label"])

    dataset_label_count[label] += 1

    basic = normalize_basic(text)
    loose = normalize_loose(text)

    record = {
        "idx": idx,
        "text": text,
        "label": label,
        "label_text": LABEL_NAMES.get(label, "Unknown"),
        "basic": basic,
        "loose": loose,
        "length": len(text)
    }

    train_records.append(record)
    raw_train_map[text] = record
    basic_train_map[basic] = record
    loose_train_map[loose] = record

print(f"训练集样本数: {len(train_records)}")
print("训练集类别分布:")
for label in sorted(dataset_label_count.keys()):
    print(f"  {LABEL_NAMES.get(label, 'Unknown')} (label={label}): {dataset_label_count[label]} 条")


# ============================================================
# 建立简单倒排索引，加速 fuzzy 匹配
# ============================================================

print("=" * 60)
print("Building token index")
print("=" * 60)

token_to_train_ids = defaultdict(set)

for rec in train_records:
    tokens = set(re.findall(r"[a-z0-9]{4,}", rec["loose"]))

    for tok in tokens:
        token_to_train_ids[tok].add(rec["idx"])

print(f"倒排索引 token 数: {len(token_to_train_ids)}")


def get_candidate_train_records(candidate_text, top_k=1000):
    """
    根据 token overlap 选出可能相似的训练样本，再做 fuzzy 精排。
    """
    cand_loose = normalize_loose(candidate_text)
    cand_tokens = set(re.findall(r"[a-z0-9]{4,}", cand_loose))

    if not cand_tokens:
        return train_records[:top_k]

    score_counter = Counter()

    for tok in cand_tokens:
        for train_id in token_to_train_ids.get(tok, []):
            score_counter[train_id] += 1

    if not score_counter:
        return train_records[:top_k]

    candidate_ids = [idx for idx, _ in score_counter.most_common(top_k)]

    return [train_records[idx] for idx in candidate_ids]


# ============================================================
# 读取生成结果（支持JSONL和TXT）
# ============================================================

print("=" * 60)
print("Loading generated candidates from file")
print("=" * 60)

candidates, candidates_metadata = load_candidates_from_file(attack_path)

if len(candidates) == 0:
    print("没有加载到有效候选样本，请检查文件格式。")
    print("支持格式：")
    print("  - JSONL: 每行一个JSON对象，包含 'text' 或 'generated' 字段")
    print("  - TXT: 格式化的文本文件，包含 [Rank] 和 Text: 字段")
    exit()


# ============================================================
# 匹配：exact / normalized exact / contains / fuzzy
# ============================================================

print("=" * 60)
print("Matching generated samples with train set")
print("=" * 60)

all_match_results = []

exact_matches = []
normalized_exact_matches = []
contains_matches = []
fuzzy_matches = []
unmatched = []

for cand_id, cand in enumerate(candidates):
    cand_raw = cand.strip()
    cand_basic = normalize_basic(cand_raw)
    cand_loose = normalize_loose(cand_raw)

    matched = False
    
    # 获取候选的元数据
    metadata = candidates_metadata[cand_id] if cand_id < len(candidates_metadata) else {}

    # --------------------------------------------------------
    # 1. raw exact
    # --------------------------------------------------------
    if cand_raw in raw_train_map:
        rec = raw_train_map[cand_raw]

        result = {
            "candidate_id": cand_id,
            "match_type": "raw_exact",
            "score": 1.0,
            "generated": cand_raw,
            "gold": rec["text"],
            "train_idx": rec["idx"],
            "label": rec["label"],
            "label_text": rec["label_text"],
            "generated_length": len(cand_raw),
            "gold_length": rec["length"],
            "metadata": metadata
        }

        exact_matches.append(result)
        all_match_results.append(result)
        matched = True

    # --------------------------------------------------------
    # 2. normalized exact
    # --------------------------------------------------------
    if not matched and cand_basic in basic_train_map:
        rec = basic_train_map[cand_basic]

        result = {
            "candidate_id": cand_id,
            "match_type": "normalized_exact",
            "score": 1.0,
            "generated": cand_raw,
            "gold": rec["text"],
            "train_idx": rec["idx"],
            "label": rec["label"],
            "label_text": rec["label_text"],
            "generated_length": len(cand_raw),
            "gold_length": rec["length"],
            "metadata": metadata
        }

        normalized_exact_matches.append(result)
        all_match_results.append(result)
        matched = True

    # --------------------------------------------------------
    # 3. contains match
    # --------------------------------------------------------
    if not matched:
        possible_records = get_candidate_train_records(cand_raw, TOP_K_CANDIDATES)

        for rec in possible_records:
            train_loose = rec["loose"]

            if len(cand_loose) >= CONTAINS_MIN_LEN and cand_loose in train_loose:
                result = {
                    "candidate_id": cand_id,
                    "match_type": "generated_is_substring_of_train",
                    "score": 1.0,
                    "generated": cand_raw,
                    "gold": rec["text"],
                    "train_idx": rec["idx"],
                    "label": rec["label"],
                    "label_text": rec["label_text"],
                    "generated_length": len(cand_raw),
                    "gold_length": rec["length"],
                    "metadata": metadata
                }

                contains_matches.append(result)
                all_match_results.append(result)
                matched = True
                break

            if len(train_loose) >= CONTAINS_MIN_LEN and train_loose in cand_loose:
                result = {
                    "candidate_id": cand_id,
                    "match_type": "train_is_substring_of_generated",
                    "score": 1.0,
                    "generated": cand_raw,
                    "gold": rec["text"],
                    "train_idx": rec["idx"],
                    "label": rec["label"],
                    "label_text": rec["label_text"],
                    "generated_length": len(cand_raw),
                    "gold_length": rec["length"],
                    "metadata": metadata
                }

                contains_matches.append(result)
                all_match_results.append(result)
                matched = True
                break

    # --------------------------------------------------------
    # 4. fuzzy match
    # --------------------------------------------------------
    if not matched:
        possible_records = get_candidate_train_records(cand_raw, TOP_K_CANDIDATES)

        best_score = 0.0
        best_rec = None

        for rec in possible_records:
            score = best_similarity(cand_raw, rec["text"])

            if score > best_score:
                best_score = score
                best_rec = rec

        if best_rec is not None and best_score >= FUZZY_THRESHOLD:
            result = {
                "candidate_id": cand_id,
                "match_type": "fuzzy",
                "score": float(best_score),
                "generated": cand_raw,
                "gold": best_rec["text"],
                "train_idx": best_rec["idx"],
                "label": best_rec["label"],
                "label_text": best_rec["label_text"],
                "generated_length": len(cand_raw),
                "gold_length": best_rec["length"],
                "metadata": metadata
            }

            fuzzy_matches.append(result)
            all_match_results.append(result)
            matched = True

    # --------------------------------------------------------
    # 5. unmatched
    # --------------------------------------------------------
    if not matched:
        unmatched.append({
            "candidate_id": cand_id,
            "generated": cand_raw,
            "generated_length": len(cand_raw),
            "metadata": metadata
        })

    if (cand_id + 1) % 50 == 0:
        print(f"已处理 {cand_id + 1}/{len(candidates)} 条候选样本")


# ============================================================
# 统计
# ============================================================

print("\n" + "=" * 60)
print("Soft Extraction Test Summary")
print("=" * 60)

print(f"候选生成样本数: {len(candidates)}")
print(f"Raw exact matches: {len(exact_matches)}")
print(f"Normalized exact matches: {len(normalized_exact_matches)}")
print(f"Contains matches: {len(contains_matches)}")
print(f"Fuzzy matches >= {FUZZY_THRESHOLD}: {len(fuzzy_matches)}")
print(f"Total matched: {len(all_match_results)}")
print(f"Unmatched: {len(unmatched)}")

if len(candidates) > 0:
    print(f"总匹配率: {len(all_match_results) / len(candidates):.2%}")

match_label_count = defaultdict(int)
match_type_count = defaultdict(int)

for item in all_match_results:
    match_label_count[item["label"]] += 1
    match_type_count[item["match_type"]] += 1

print("\n匹配类型分布:")
for k, v in sorted(match_type_count.items()):
    print(f"  {k}: {v}")

print("\n匹配类别分布:")
for label in sorted(match_label_count.keys()):
    print(f"  {LABEL_NAMES.get(label, 'Unknown')} (label={label}): {match_label_count[label]}")


# ============================================================
# 保存结果
# ============================================================

json_path = os.path.join(save_dir, f"soft_matches_{timestamp}.json")
csv_path = os.path.join(save_dir, f"soft_matches_{timestamp}.csv")
txt_path = os.path.join(save_dir, f"soft_matches_{timestamp}.txt")
summary_path = os.path.join(save_dir, f"summary_{timestamp}.txt")
unmatched_path = os.path.join(save_dir, f"unmatched_{timestamp}.txt")

# JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_match_results, f, ensure_ascii=False, indent=2)

# CSV
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "candidate_id",
        "match_type",
        "score",
        "train_idx",
        "label",
        "label_text",
        "generated_length",
        "gold_length",
        "prompt",
        "ppl",
        "zlib",
        "ratio",
        "ppl_zlib_ratio",
        "generated",
        "gold"
    ])

    for item in all_match_results:
        metadata = item.get("metadata", {})
        writer.writerow([
            item["candidate_id"],
            item["match_type"],
            f"{item['score']:.4f}",
            item["train_idx"],
            item["label"],
            item["label_text"],
            item["generated_length"],
            item["gold_length"],
            metadata.get("prompt", ""),
            metadata.get("ppl", ""),
            metadata.get("zlib", ""),
            metadata.get("ratio", ""),
            metadata.get("ppl_zlib_ratio", ""),
            item["generated"],
            item["gold"]
        ])

# TXT 详细结果
all_match_results_sorted = sorted(
    all_match_results,
    key=lambda x: (-x["score"], x["match_type"], x["candidate_id"])
)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("SST-2 训练数据抽取软匹配结果\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"候选生成样本数: {len(candidates)}\n")
    f.write(f"Raw exact matches: {len(exact_matches)}\n")
    f.write(f"Normalized exact matches: {len(normalized_exact_matches)}\n")
    f.write(f"Contains matches: {len(contains_matches)}\n")
    f.write(f"Fuzzy matches >= {FUZZY_THRESHOLD}: {len(fuzzy_matches)}\n")
    f.write(f"Total matched: {len(all_match_results)}\n")
    f.write(f"Unmatched: {len(unmatched)}\n")
    f.write(f"总匹配率: {len(all_match_results) / len(candidates):.2%}\n\n")

    f.write("=" * 80 + "\n")
    f.write("匹配详情\n")
    f.write("=" * 80 + "\n\n")

    for i, item in enumerate(all_match_results_sorted, 1):
        metadata = item.get("metadata", {})
        f.write(f"[{i}] Candidate ID: {item['candidate_id']}\n")
        f.write(f"Match Type: {item['match_type']}\n")
        f.write(f"Score: {item['score']:.4f}\n")
        f.write(f"Train Index: {item['train_idx']}\n")
        f.write(f"Label: {item['label']} ({item['label_text']})\n")
        f.write(f"Generated Length: {item['generated_length']}\n")
        f.write(f"Gold Length: {item['gold_length']}\n")
        if metadata:
            f.write(f"Prompt: {metadata.get('prompt', 'N/A')}\n")
            f.write(f"PPL: {metadata.get('ppl', 'N/A')}\n")
            f.write(f"ZLIB: {metadata.get('zlib', 'N/A')}\n")
            f.write(f"Ratio: {metadata.get('ratio', 'N/A')}\n")
            f.write(f"PPL/Zlib Ratio: {metadata.get('ppl_zlib_ratio', 'N/A')}\n")
        f.write("\n")

        f.write("[Generated]\n")
        f.write(item["generated"] + "\n\n")

        f.write("[Gold]\n")
        f.write(item["gold"] + "\n\n")

        f.write("-" * 80 + "\n\n")

# summary
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("SST-2 训练数据抽取软匹配摘要\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"训练集路径: {train_dataset_path}\n")
    f.write(f"攻击结果文件: {attack_path}\n")
    f.write(f"Fuzzy threshold: {FUZZY_THRESHOLD}\n")
    f.write(f"Contains min length: {CONTAINS_MIN_LEN}\n")
    f.write(f"Top-K fuzzy candidates: {TOP_K_CANDIDATES}\n\n")

    f.write(f"训练集样本数: {len(train_records)}\n")
    f.write(f"候选生成样本数: {len(candidates)}\n")
    f.write(f"Raw exact matches: {len(exact_matches)}\n")
    f.write(f"Normalized exact matches: {len(normalized_exact_matches)}\n")
    f.write(f"Contains matches: {len(contains_matches)}\n")
    f.write(f"Fuzzy matches >= {FUZZY_THRESHOLD}: {len(fuzzy_matches)}\n")
    f.write(f"Total matched: {len(all_match_results)}\n")
    f.write(f"Unmatched: {len(unmatched)}\n")

    if len(candidates) > 0:
        f.write(f"总匹配率: {len(all_match_results) / len(candidates):.2%}\n")

    f.write("\n匹配类型分布:\n")
    for k, v in sorted(match_type_count.items()):
        f.write(f"  - {k}: {v}\n")

    f.write("\n匹配类别分布:\n")
    for label in sorted(match_label_count.keys()):
        f.write(f"  - {LABEL_NAMES.get(label, 'Unknown')} (label={label}): {match_label_count[label]}\n")

# unmatched
with open(unmatched_path, "w", encoding="utf-8") as f:
    f.write(f"未匹配候选样本，共 {len(unmatched)} 条\n")
    f.write("=" * 80 + "\n\n")

    for item in unmatched:
        metadata = item.get("metadata", {})
        f.write(f"Candidate ID: {item['candidate_id']}\n")
        f.write(f"Length: {item['generated_length']}\n")
        if metadata:
            f.write(f"Prompt: {metadata.get('prompt', 'N/A')}\n")
            f.write(f"PPL: {metadata.get('ppl', 'N/A')}\n")
            f.write(f"ZLIB: {metadata.get('zlib', 'N/A')}\n")
            f.write(f"Ratio: {metadata.get('ratio', 'N/A')}\n")
            f.write(f"PPL/Zlib Ratio: {metadata.get('ppl_zlib_ratio', 'N/A')}\n")
        f.write(item["generated"] + "\n\n")
        f.write("-" * 80 + "\n\n")


print("\n" + "=" * 60)
print("Saved files")
print("=" * 60)
print(f"JSON: {json_path}")
print(f"CSV: {csv_path}")
print(f"TXT: {txt_path}")
print(f"Summary: {summary_path}")
print(f"Unmatched: {unmatched_path}")

print("\nTop matched examples:")
for i, item in enumerate(all_match_results_sorted[:5], 1):
    print("\n" + "=" * 80)
    print(f"Rank {i}")
    print(f"Type: {item['match_type']}, Score: {item['score']:.4f}")
    print(f"Label: {item['label_text']}, Train idx: {item['train_idx']}")
    if item.get("metadata"):
        metadata = item['metadata']
        print(f"Prompt: {metadata.get('prompt', 'N/A')}")
        print(f"PPL: {metadata.get('ppl', 'N/A')}")
        print(f"PPL/Zlib: {metadata.get('ppl_zlib_ratio', 'N/A')}")
    print("[Generated]")
    print(item["generated"][:500])
    print("[Gold]")
    print(item["gold"][:500])