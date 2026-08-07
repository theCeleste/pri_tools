#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import difflib
import numpy as np
from typing import List


def read_sst2_tsv(path: str) -> pd.DataFrame:
    """
    读取 SST-2 格式的 tsv 文件，假设有 header: sentence, label
    label 只保留，不参与后续相似度计算。
    """
    df = pd.read_csv(path, sep="\t", header=0)
    # 确保有 sentence / label 列
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} 必须包含 'sentence' 和 'label' 两列")
    # 只留我们关心的两列，index 重新排
    return df[["sentence", "label"]].reset_index(drop=True)


def char_similarity(a: str, b: str) -> float:
    """
    字符级相似度，使用 difflib.SequenceMatcher 的 ratio。
    值域 [0,1]，1 表示完全相同，0 表示完全不同。
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_change_ratio(a: str, b: str) -> float:
    """
    词级改动比例（position-aware）：
    - 先按空格切词
    - 以较长句子的长度为基准，对齐位置上不同 token 的比例
    - 超出的部分按“不同”计

    返回 [0,1]，1 表示完全不同，0 表示完全相同。
    """
    a_tokens = a.split()
    b_tokens = b.split()

    max_len = max(len(a_tokens), len(b_tokens))
    if max_len == 0:
        return 0.0

    diff_count = 0
    for i in range(max_len):
        tok_a = a_tokens[i] if i < len(a_tokens) else None
        tok_b = b_tokens[i] if i < len(b_tokens) else None
        if tok_a != tok_b:
            diff_count += 1

    return diff_count / max_len


def jaccard_similarity(a: str, b: str) -> float:
    """
    词集合的 Jaccard 相似度：
    |A ∩ B| / |A ∪ B|
    """
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union) if len(union) > 0 else 0.0


def analyze_diff(orig_df: pd.DataFrame, prot_df: pd.DataFrame, max_examples: int = 5):
    """
    对两份 DataFrame (原始 & 保护后) 做对比分析。
    只比较 sentence，不比较 label。
    """
    if len(orig_df) != len(prot_df):
        raise ValueError(f"行数不一致：orig={len(orig_df)}, prot={len(prot_df)}")

    n = len(orig_df)

    char_sims: List[float] = []
    token_ratios: List[float] = []
    jacc_sims: List[float] = []
    unchanged_indices: List[int] = []

    for i in range(n):
        s_orig = str(orig_df.loc[i, "sentence"])
        s_prot = str(prot_df.loc[i, "sentence"])

        if s_orig == s_prot:
            unchanged_indices.append(i)

        c_sim = char_similarity(s_orig, s_prot)
        t_ratio = token_change_ratio(s_orig, s_prot)
        j_sim = jaccard_similarity(s_orig, s_prot)

        char_sims.append(c_sim)
        token_ratios.append(t_ratio)
        jacc_sims.append(j_sim)

    char_sims = np.array(char_sims)
    token_ratios = np.array(token_ratios)
    jacc_sims = np.array(jacc_sims)

    print("=" * 80)
    print("PTCDS 保护前后数据集内容变化幅度分析（只比较 sentence 列）")
    print("=" * 80)
    print(f"样本总数: {n}")
    print(f"完全未改变的句子数量: {len(unchanged_indices)} "
          f"({len(unchanged_indices) / n * 100:.2f}%)\n")

    print("【字符级相似度】(1=完全相同)")
    print(f"  平均值: {char_sims.mean():.4f}")
    print(f"  中位数: {np.median(char_sims):.4f}")
    print(f"  最小值: {char_sims.min():.4f}")
    print(f"  最大值: {char_sims.max():.4f}\n")

    print("【词级改动比例】(1=完全不同, 0=完全相同)")
    print(f"  平均值: {token_ratios.mean():.4f}")
    print(f"  中位数: {np.median(token_ratios):.4f}")
    print(f"  最小值: {token_ratios.min():.4f}")
    print(f"  最大值: {token_ratios.max():.4f}\n")

    print("【Jaccard 相似度】(基于词集合, 1=完全相同)")
    print(f"  平均值: {jacc_sims.mean():.4f}")
    print(f"  中位数: {np.median(jacc_sims):.4f}")
    print(f"  最小值: {jacc_sims.min():.4f}")
    print(f"  最大值: {jacc_sims.max():.4f}\n")

    # 打印变化最大的几个样本
    print("=" * 80)
    print("变化最大的若干样本（按词级改动比例排序）")
    print("=" * 80)
    top_k = min(max_examples, n)
    worst_idx = np.argsort(-token_ratios)[:top_k]
    for rank, idx in enumerate(worst_idx, start=1):
        print(f"\n[变化第 {rank} 大] 样本索引: {idx}")
        print(f"  词级改动比例: {token_ratios[idx]:.4f}")
        print(f"  字符级相似度: {char_sims[idx]:.4f}")
        print(f"  Jaccard 相似度: {jacc_sims[idx]:.4f}")
        print("  原句:", orig_df.loc[idx, "sentence"])
        print("  保护后:", prot_df.loc[idx, "sentence"])

    print("\n" + "=" * 80)
    print("变化最小但非零的若干样本（同样只看 sentence）")
    print("=" * 80)
    changed_indices = np.where(token_ratios > 0)[0]
    if len(changed_indices) > 0:
        best_idx = changed_indices[np.argsort(token_ratios[changed_indices])[:top_k]]
        for rank, idx in enumerate(best_idx, start=1):
            print(f"\n[变化第 {rank} 小] 样本索引: {idx}")
            print(f"  词级改动比例: {token_ratios[idx]:.4f}")
            print(f"  字符级相似度: {char_sims[idx]:.4f}")
            print(f"  Jaccard 相似度: {jacc_sims[idx]:.4f}")
            print("  原句:", orig_df.loc[idx, "sentence"])
            print("  保护后:", prot_df.loc[idx, "sentence"])
    else:
        print("  没有发生任何变化的样本。")


def main():
    parser = argparse.ArgumentParser(
        description="对比 PTCDS 保护前后 SST-2 数据集内容改变幅度（只比较 sentence）"
    )
    parser.add_argument(
        "--orig",
        type=str,
        default="/home/wpj/ptcds/PTCDS/data/SST-2/train.tsv",
        help="原始 SST-2 train.tsv 路径",
    )
    parser.add_argument(
        "--prot",
        type=str,
        default="/home/wpj/ptcds/PTCDS/output_PTCDS/SST-2_glove_eps4/eps_4.00/sword_0.50_p_0.20/train.tsv",
        help="PTCDS 保护后的 train.tsv 路径",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="打印变化最大/最小的样本个数",
    )

    args = parser.parse_args()

    print(f"加载原始数据集: {args.orig}")
    orig_df = read_sst2_tsv(args.orig)
    print(f"加载保护后数据集: {args.prot}")
    prot_df = read_sst2_tsv(args.prot)

    analyze_diff(orig_df, prot_df, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
