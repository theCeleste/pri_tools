"""
公平对比有隐私 / 无隐私模型的 MIA 攻击性能

核心思想（更贴近“防御场景公平性”）：
1. 先只用「无隐私保护模型」的结果，在给定目标 FPR（1%, 0.1%, 0.01% ...）下
   搜索出对应的阈值 γ_baseline。
2. 固定这些阈值 γ_baseline，再分别在：
   - 无隐私模型上计算 (TPR_base, FPR_base)
   - 有隐私模型上计算 (TPR_priv, FPR_priv)
   然后对比两者。
3. 支持控制正负样本比例（训练=正例，测试=负例）：
   - 1:1（更标准的评估）
   - 其他比例（例如 1:100，更贴近现实）

使用示例：
python evaluate_mia_fair_compare.py \
    --base-train-pkl wo_all_scores_sst2_roberta_1_train.pkl \
    --base-test-pkl  wo_all_scores_sst2_roberta_1_test.pkl \
    --priv-train-pkl all_scores_sst2_roberta_1_train.pkl \
    --priv-test-pkl  all_scores_sst2_roberta_1_test.pkl \
    --target-fprs 0.01 0.001 0.0001 \
    --pos-neg-ratio 1.0
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from typing import Tuple

from read_neighbour_scores import analyze_single_text


def extract_score_gaps(all_scores):
    """从pkl文件中提取所有样本的score_gap"""
    score_gaps = []

    for scores_dict in all_scores:
        analysis = analyze_single_text(scores_dict)
        if analysis is not None:
            score_gaps.append(analysis["score_gap"])
        else:
            score_gaps.append(0.0)

    return np.array(score_gaps)


def subsample_with_ratio(
    score_gaps_train: np.ndarray,
    score_gaps_test: np.ndarray,
    pos_neg_ratio: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按指定正负样本比例（train:test）进行子采样。

    Args:
        score_gaps_train: 训练集score_gap（正例）
        score_gaps_test:  测试集score_gap（负例）
        pos_neg_ratio:    正:负 比例。例如 1.0 表示 1:1，0.01 表示 1:100
        seed:             随机种子

    Returns:
        sub_train, sub_test: 采样后的训练/测试 score_gap
    """
    rng = np.random.RandomState(seed)

    n_pos_max = len(score_gaps_train)
    n_neg_max = len(score_gaps_test)

    # 如果 ratio = P/N，则 N = min(n_neg_max, n_pos_max / ratio)
    # 再 P = min(n_pos_max, round(N * ratio))
    if pos_neg_ratio <= 0:
        raise ValueError("pos_neg_ratio 必须 > 0")

    # 先决定负例数
    # N <= n_neg_max 且 N <= n_pos_max / ratio
    n_neg_by_pos = int(n_pos_max / pos_neg_ratio)
    n_neg = min(n_neg_max, n_neg_by_pos)
    if n_neg <= 0:
        raise ValueError("根据当前样本数和比例，无法采样到任何负例，请检查比例或数据量。")

    # 再算正例数
    n_pos = int(round(n_neg * pos_neg_ratio))
    n_pos = min(n_pos, n_pos_max)
    if n_pos <= 0:
        raise ValueError("根据当前样本数和比例，无法采样到任何正例，请检查比例或数据量。")

    # 下标随机采样
    pos_idx = rng.choice(len(score_gaps_train), size=n_pos, replace=False)
    neg_idx = rng.choice(len(score_gaps_test), size=n_neg, replace=False)

    return score_gaps_train[pos_idx], score_gaps_test[neg_idx]


def calculate_tpr_fpr_at_threshold(
    score_gaps_train: np.ndarray,
    score_gaps_test: np.ndarray,
    threshold: float,
) -> Tuple[float, float, int, int, int, int]:
    """在给定阈值下计算TPR和FPR"""
    # 预测：score_gap > threshold 则判断为成员
    pred_train = (score_gaps_train > threshold).astype(int)
    pred_test = (score_gaps_test > threshold).astype(int)

    # 真实标签：训练集=1（成员），测试集=0（非成员）
    true_train = np.ones_like(pred_train)
    true_test = np.zeros_like(pred_test)

    y_true = np.concatenate([true_train, true_test])
    y_pred = np.concatenate([pred_train, pred_test])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr, tp, fp, tn, fn


def find_threshold_on_baseline(
    score_gaps_train_base: np.ndarray,
    score_gaps_test_base: np.ndarray,
    target_fpr: float,
    num_steps: int = 10000,
) -> Tuple[float, float, float]:
    """
    在「无隐私保护模型」上，根据目标FPR搜索阈值。
    返回：threshold, tpr_base, fpr_base
    """
    all_score_gaps = np.concatenate([score_gaps_train_base, score_gaps_test_base])
    min_gap = float(all_score_gaps.min())
    max_gap = float(all_score_gaps.max())

    thresholds = np.linspace(min_gap, max_gap, num_steps)

    best_threshold = None
    best_tpr = 0.0
    best_fpr = 0.0
    min_fpr_diff = float("inf")

    for th in thresholds:
        tpr, fpr, _, _, _, _ = calculate_tpr_fpr_at_threshold(
            score_gaps_train_base, score_gaps_test_base, th
        )
        fpr_diff = abs(fpr - target_fpr)
        if fpr <= target_fpr and fpr_diff < min_fpr_diff:
            min_fpr_diff = fpr_diff
            best_threshold = float(th)
            best_tpr = float(tpr)
            best_fpr = float(fpr)

    # 如果所有阈值的FPR都大于target_fpr，就取最小gap作为阈值
    if best_threshold is None:
        best_threshold = min_gap
        best_tpr, best_fpr, _, _, _, _ = calculate_tpr_fpr_at_threshold(
            score_gaps_train_base, score_gaps_test_base, best_threshold
        )
        best_tpr = float(best_tpr)
        best_fpr = float(best_fpr)

    return best_threshold, best_tpr, best_fpr


def fair_compare(
    base_train_pkl: str,
    base_test_pkl: str,
    priv_train_pkl: str,
    priv_test_pkl: str,
    target_fprs,
    pos_neg_ratio: float = 1.0,
    seed: int = 0,
):
    """
    公平对比有隐私 / 无隐私模型：
    1. 用无隐私模型在目标FPR下找到阈值
    2. 用同一阈值评估两个模型的TPR与FPR
    3. 训练/测试样本按给定比例子采样
    """
    print("=" * 80)
    print("公平 MIA 评估（先在无隐私模型上定阈值，再对比两模型）")
    print("=" * 80)

    print("\n[1] 加载无隐私保护模型结果（baseline）...")
    with open(base_train_pkl, "rb") as f:
        base_train_scores = pickle.load(f)
    with open(base_test_pkl, "rb") as f:
        base_test_scores = pickle.load(f)
    print(f"  baseline 训练样本数: {len(base_train_scores)}")
    print(f"  baseline 测试样本数: {len(base_test_scores)}")

    print("\n[2] 加载有隐私保护模型结果（protected）...")
    with open(priv_train_pkl, "rb") as f:
        priv_train_scores = pickle.load(f)
    with open(priv_test_pkl, "rb") as f:
        priv_test_scores = pickle.load(f)
    print(f"  protected 训练样本数: {len(priv_train_scores)}")
    print(f"  protected 测试样本数: {len(priv_test_scores)}")

    # 提取 score_gap
    print("\n[3] 提取 score_gap ...")
    base_train_gaps = extract_score_gaps(base_train_scores)
    base_test_gaps = extract_score_gaps(base_test_scores)
    priv_train_gaps = extract_score_gaps(priv_train_scores)
    priv_test_gaps = extract_score_gaps(priv_test_scores)

    print(f"\n  baseline 训练集均值: {base_train_gaps.mean():.4f}, 测试集均值: {base_test_gaps.mean():.4f}")
    print(f"  protected 训练集均值: {priv_train_gaps.mean():.4f}, 测试集均值: {priv_test_gaps.mean():.4f}")

    # 子采样以控制正负样本比例（同时对子模型使用相同的采样大小）
    print(f"\n[4] 按正:负 = {pos_neg_ratio}:1 进行子采样（平衡评估数据集）...")
    base_train_sub, base_test_sub = subsample_with_ratio(
        base_train_gaps, base_test_gaps, pos_neg_ratio, seed=seed
    )
    # 为了公平，对protected也用同样的采样规模（但独立随机）
    priv_train_sub, priv_test_sub = subsample_with_ratio(
        priv_train_gaps, priv_test_gaps, pos_neg_ratio, seed=seed + 1
    )

    print(f"  baseline:  训练{len(base_train_sub)} / 测试{len(base_test_sub)}")
    print(f"  protected: 训练{len(priv_train_sub)} / 测试{len(priv_test_sub)}")

    print("\n[5] 在 baseline 上搜索阈值，并用同一阈值评估两模型 ...")

    records = []

    for target_fpr in target_fprs:
        print("\n" + "-" * 80)
        print(f"目标 FPR (baseline): {target_fpr * 100:.2f}%")

        # 1) 只在 baseline 上搜索阈值
        th, tpr_base, fpr_base = find_threshold_on_baseline(
            base_train_sub, base_test_sub, target_fpr
        )

        print(f"  baseline 阈值 γ: {th:.6f}")
        print(f"  baseline 实际 FPR: {fpr_base * 100:.4f}%")
        print(f"  baseline TPR:      {tpr_base * 100:.4f}%")

        # 2) 用同一阈值评估 baseline & protected
        _, _, tp_b, fp_b, tn_b, fn_b = calculate_tpr_fpr_at_threshold(
            base_train_sub, base_test_sub, th
        )
        tpr_p, fpr_p, tp_p, fp_p, tn_p, fn_p = calculate_tpr_fpr_at_threshold(
            priv_train_sub, priv_test_sub, th
        )

        print(f"\n  使用同一阈值 γ 在 protected 上评估：")
        print(f"    protected FPR: {fpr_p * 100:.4f}%")
        print(f"    protected TPR: {tpr_p * 100:.4f}%")

        records.append(
            {
                "target_fpr": target_fpr,
                "target_fpr_percent": f"{target_fpr * 100:.2f}%",
                "threshold_from_baseline": th,
                # baseline
                "baseline_tpr": tpr_base,
                "baseline_tpr_percent": f"{tpr_base * 100:.4f}%",
                "baseline_fpr": fpr_base,
                "baseline_fpr_percent": f"{fpr_base * 100:.4f}%",
                "baseline_tp": tp_b,
                "baseline_fp": fp_b,
                "baseline_tn": tn_b,
                "baseline_fn": fn_b,
                # protected
                "protected_tpr": tpr_p,
                "protected_tpr_percent": f"{tpr_p * 100:.4f}%",
                "protected_fpr": fpr_p,
                "protected_fpr_percent": f"{fpr_p * 100:.4f}%",
                "protected_tp": tp_p,
                "protected_fp": fp_p,
                "protected_tn": tn_p,
                "protected_fn": fn_p,
                # 采样信息
                "pos_neg_ratio": pos_neg_ratio,
                "baseline_pos_samples": len(base_train_sub),
                "baseline_neg_samples": len(base_test_sub),
                "protected_pos_samples": len(priv_train_sub),
                "protected_neg_samples": len(priv_test_sub),
            }
        )

    df = pd.DataFrame(records)
    #保存文件，结果
    out_file = "mia_fair_comparison_results.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("公平比较结果（摘要）")
    print("=" * 80)
    cols_show = [
        "target_fpr_percent",
        "threshold_from_baseline",
        "baseline_fpr_percent",
        "baseline_tpr_percent",
        "protected_fpr_percent",
        "protected_tpr_percent",
    ]
    print(df[cols_show].to_string(index=False))
    print(f"\n详细结果已保存到: {out_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="公平对比有/无隐私模型的MIA攻击性能（阈值由无隐私模型确定）"
    )
    parser.add_argument(
        "--base-train-pkl", type=str, required=True, help="无隐私模型的训练集 pkl"
    )
    parser.add_argument(
        "--base-test-pkl", type=str, required=True, help="无隐私模型的测试集 pkl"
    )
    parser.add_argument(
        "--priv-train-pkl", type=str, required=True, help="有隐私模型的训练集 pkl"
    )
    parser.add_argument(
        "--priv-test-pkl", type=str, required=True, help="有隐私模型的测试集 pkl"
    )
    parser.add_argument(
        "--target-fprs",
        type=float,
        nargs="+",
        default=[0.01, 0.001, 0.0001],
        help="目标 FPR 列表（例如 0.01 0.001 0.0001）",
    )
    parser.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=1.0,
        help="评估时使用的正:负 样本比例（train:test），例如 1.0 表示 1:1，0.01 表示 1:100",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="子采样随机种子（控制可复现性）"
    )

    args = parser.parse_args()

    fair_compare(
        base_train_pkl=args.base_train_pkl,
        base_test_pkl=args.base_test_pkl,
        priv_train_pkl=args.priv_train_pkl,
        priv_test_pkl=args.priv_test_pkl,
        target_fprs=args.target_fprs,
        pos_neg_ratio=args.pos_neg_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


