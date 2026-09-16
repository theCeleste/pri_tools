#!/usr/bin/env python3
"""
后处理脚本：计算攻击性能指标并绘制 ROC 曲线
适用于 attack_results_*.csv 文件
"""
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def epsilon_empirical(FPR, FNR, n):
    """
    Compute empirical epsilon from FPR, FNR, and dataset size n.
    δ = 1 / (5n)
    当FNR或FPR为零时，对应的eps值设为float('inf')
    """
    delta = 1 / (5 * n)
    
    # 初始化eps值
    eps1 = 0.0
    eps2 = 0.0
    
    # 计算eps1: 当FNR为0时设为inf，否则计算对数
    if FNR == 0:
        eps1 = float('inf')
    else:
        # 确保分母和分子都有效
        if (1 - delta - FPR) > 0:
            eps1 = math.log((1 - delta - FPR) / FNR)
        else:
            raise ValueError("Invalid input: (1 - delta - FPR) must be positive")
    
    # 计算eps2: 当FPR为0时设为inf，否则计算对数
    if FPR == 0:
        eps2 = float('inf')
    else:
        # 确保分母和分子都有效
        if (1 - delta - FNR) > 0:
            eps2 = math.log((1 - delta - FNR) / FPR)
        else:
            raise ValueError("Invalid input: (1 - delta - FNR) must be positive")
    
    return max(eps1, eps2)

def compute_metrics(csv_file, csv_file_nonmember=None, positive_ratio=None, plot=True):
    """
    更稳健的评估函数。
    用法（优先级）:
      1) 如果 csv_file 含 'label'（0/1）列 -> 直接用它
      2) 否则，如果提供 csv_file_nonmember -> 将两个文件 concat，并给标签 (1,0)
      3) 否则，如果提供 positive_ratio -> 按比例在 csv 中构造（不推荐，需谨慎）
      4) 否则 抛错

    返回: dict(AUC, TPR@1%, TPR@0.1%)
    """
    df = pd.read_csv(csv_file)
    if 'diff_score' not in df.columns:
        raise ValueError("CSV must contain 'diff_score' column")

    # 情况一：CSV 自带 label
    if 'label' in df.columns or 'is_member' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'is_member'
        if set(df[label_col].unique()) - {0,1}:
            raise ValueError(f"Label column {label_col} must be 0/1")
        y_true = df[label_col].astype(int).values
        y_score = df['diff_score'].values

    # 情况二：提供单独的非成员文件
    elif csv_file_nonmember is not None:
        df_pos = df.copy()
        df_pos['label'] = 1
        df_neg = pd.read_csv(csv_file_nonmember)
        if 'diff_score' not in df_neg.columns:
            raise ValueError("nonmember csv must contain 'diff_score' column")
        df_neg['label'] = 0
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        y_true = df_all['label'].values
        y_score = df_all['diff_score'].values

    # 情况三：按比例构造（不安全，谨慎使用）
    elif positive_ratio is not None:
        n = len(df)
        n_members = int(n * positive_ratio)
        if n_members == 0 or n_members == n:
            raise ValueError("positive_ratio leads to degenerate class sizes")
        # **重要**：这里不对 df 做排序，直接按照文件顺序切分（但这只有在你知道文件顺序与分割策略一致时可用）
        y_true = np.array([1]*n_members + [0]*(n - n_members))
        y_score = df['diff_score'].values[:n]  # 注意：确保文件顺序为先 members 再 nonmembers
        if len(y_score) != len(y_true):
            raise ValueError("length mismatch")
    else:
        raise ValueError("No way to determine labels. Provide 'label' column, or csv_file_nonmember, or positive_ratio")

    # Sanity checks
    if np.isnan(y_score).any():
        raise ValueError("Found NaN in scores")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class in y_true. Need both members and non-members to compute ROC/AUC")

    # Compute ROC/AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    # get TPR at target FPR (interpolate conservatively)
    def get_tpr_at_fpr(target_fpr):
        # find the max TPR where FPR <= target_fpr
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) == 0:
            return 0.0
        return float(tpr[idx[-1]])

    tpr_1 = get_tpr_at_fpr(0.01)
    tpr_01 = get_tpr_at_fpr(0.001)
    
    # 计算FNR (False Negative Rate) = 1 - TPR
    fnr_1 = 1 - tpr_1
    fnr_01 = 1 - tpr_01

    # epsilon = epsilon_empirical(0.01, fnr_1, 5)
    # print(f"Empirical epsilon ≈ {epsilon:.3f}")

    # print(f"AUC = {auc_val:.4f}")
    # print(f"TPR@1%FPR = {tpr_1*100:.2f}%")
    # # print(f"FNR@1%FPR = {fnr_1*100:.2f}%")
    # print(f"TPR@0.1%FPR = {tpr_01*100:.2f}%")
    # # print(f"FNR@0.1%FPR = {fnr_01*100:.2f}%")

    if plot:
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
        plt.plot([0,1],[0,1],'k--', alpha=0.4)
        plt.xscale('log')
        plt.xlim(1e-5,1)
        plt.ylim(0,1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return {"AUC": auc_val, "TPR@1%": tpr_1, "FNR@1%": fnr_1, "TPR@0.1%": tpr_01, "FNR@0.1%": fnr_01}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="attack_results_*.csv 文件路径")
    parser.add_argument("--ratio", type=float, default=0.02, help="成员样本比例（默认 0.5）")
    args = parser.parse_args()

    compute_metrics(args.file, positive_ratio=args.ratio)
