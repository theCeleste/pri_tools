import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR.parent / "datasets" / "SST-2" / "train.tsv"
DEFAULT_ATTACK_FILES = [
    SCRIPT_DIR / "outputs" / "top_n_samples.txt",
    SCRIPT_DIR / "outputs" / "top_n_samples-sv.txt",
]


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def load_training_sentences(dataset_path):
    sentences = set()

    if dataset_path.suffix.lower() == ".tsv":
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("sentence")
                if text is None:
                    continue

                normalized = normalize_text(str(text))
                if normalized:
                    sentences.add(normalized)
        return sentences

    with open(dataset_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for item in json_data:
        text = item.get("original_text")
        if text is None:
            text = item.get("text")
        if text is None:
            text = item.get("sentence")
        if text is None:
            continue

        normalized = normalize_text(str(text))
        if normalized:
            sentences.add(normalized)

    return sentences


def parse_topn_output(attack_path):
    with open(attack_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    samples_by_metric = defaultdict(list)
    current_metric = "UNSCOPED"
    current_sample_lines = []
    collecting_sample = False

    def flush_current_sample():
        nonlocal current_sample_lines
        if not current_sample_lines:
            return
        sample = "\n".join(current_sample_lines).strip()
        current_sample_lines = []
        if sample:
            samples_by_metric[current_metric].append(sample)

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("Metric Name:"):
            flush_current_sample()
            current_metric = line.split("Metric Name:", 1)[1].strip() or "UNSCOPED"
            collecting_sample = False
            continue

        if re.match(r"^\d+:\s", line):
            flush_current_sample()
            collecting_sample = True
            continue

        if collecting_sample:
            if not line.strip() and not current_sample_lines:
                continue
            current_sample_lines.append(line)

    flush_current_sample()
    return dict(samples_by_metric)


def extract_unique_lines(samples):
    sample_lines = set()
    for sample in samples:
        for line in sample.splitlines():
            normalized = normalize_text(line)
            if normalized:
                sample_lines.add(normalized)
    return sample_lines


def evaluate_samples(samples, training_sentences):
    normalized_samples = {
        normalize_text(sample) for sample in samples if normalize_text(sample)
    }
    sample_matches = normalized_samples & training_sentences

    unique_lines = extract_unique_lines(samples)
    line_matches = unique_lines & training_sentences

    return {
        "sample_count": len(normalized_samples),
        "sample_match_count": len(sample_matches),
        "sample_match_rate": (
            len(sample_matches) / len(normalized_samples) if normalized_samples else 0.0
        ),
        "line_count": len(unique_lines),
        "line_match_count": len(line_matches),
        "line_match_rate": len(line_matches) / len(unique_lines) if unique_lines else 0.0,
        "sample_matches": sorted(sample_matches),
        "line_matches": sorted(line_matches),
    }


def print_result(title, result):
    print(title)
    print(
        f"  唯一样本数: {result['sample_count']} | "
        f"整段命中: {result['sample_match_count']} ({result['sample_match_rate']:.2%})"
    )
    print(
        f"  唯一行数: {result['line_count']} | "
        f"单行命中: {result['line_match_count']} ({result['line_match_rate']:.2%})"
    )

    preview = result["line_matches"][:5]
    if preview:
        print("  命中示例:")
        for text in preview:
            print(f"    - {text}")
    print()


def evaluate_attack_file(attack_path, training_sentences):
    samples_by_metric = parse_topn_output(attack_path)
    all_samples = []
    for samples in samples_by_metric.values():
        all_samples.extend(samples)

    print(f"===== {attack_path.name} =====")
    print_result("整体结果", evaluate_samples(all_samples, training_sentences))

    for metric_name, samples in samples_by_metric.items():
        print_result(f"Metric: {metric_name}", evaluate_samples(samples, training_sentences))


def main():
    parser = argparse.ArgumentParser(
        description="比较 top_n 输出文件中样本与训练集句子的重合情况。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="训练集路径，默认使用 datasets/SST-2/train.tsv",
    )
    parser.add_argument(
        "--attack-files",
        nargs="+",
        type=Path,
        default=DEFAULT_ATTACK_FILES,
        help="要比较的攻击输出文件列表",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    training_sentences = load_training_sentences(dataset_path)
    print(f"训练集句子数: {len(training_sentences)}")
    print(f"数据集路径: {dataset_path}")
    print()

    existing_files = []
    for attack_file in args.attack_files:
        attack_path = attack_file
        if not attack_path.is_absolute():
            cwd_candidate = attack_path.resolve()
            script_dir_candidate = (SCRIPT_DIR / attack_path).resolve()
            attack_path = cwd_candidate if cwd_candidate.exists() else script_dir_candidate
        if attack_path.exists():
            existing_files.append(attack_path)
        else:
            print(f"跳过不存在的文件: {attack_path}")

    print()

    if not existing_files:
        raise FileNotFoundError("没有找到任何可评估的攻击输出文件。")

    for attack_path in existing_files:
        evaluate_attack_file(attack_path, training_sentences)


if __name__ == "__main__":
    main()
