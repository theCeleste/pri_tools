"""
简化版：对比两个已有pkl文件的模型MIA攻击效果

如果你已经分别对两个模型运行了neighbour_attack.py，生成了pkl文件，
可以使用这个脚本直接对比，无需重新运行攻击。

使用方法：
python compare_models_mia_simple.py \
    --protected_pkl ./wo_all_eps3_20_12.11_scores_mergedsst2_bert_0.pkl \
    --baseline_pkl ./wo_all_base_20_12.11_scores_sst2_bert_0.pkl
"""

import pickle
import numpy as np
import pandas as pd
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，将跳过统计检验")
import argparse


def analyze_scores(all_scores):
    """分析攻击结果，提取分数差距"""
    score_gaps = []
    original_texts = []
    
    for scores_dict in all_scores:
        # 提取原始文本分数
        original_score = None
        original_text = None
        for key in scores_dict.keys():
            if key.startswith('<original_text>:'):
                original_score = scores_dict[key]
                original_text = key.replace('<original_text>: ', '')
                break
        
        if original_score is None:
            continue
        
        # 提取邻居文本分数
        neighbour_scores = []
        for key, score in scores_dict.items():
            # 1. 跳过原句
            if isinstance(key, str) and key.startswith('<original_text>:'):
                continue

            # 2. 第一种格式：tuple key = (邻居文本, MLM分数)
            if isinstance(key, tuple) and len(key) == 2:
                if isinstance(score, list) and len(score) > 0:
                    neighbour_scores.append(score[0])
                else:
                    neighbour_scores.append(score)

            # 3. 第二种格式：key 就是邻居句子字符串
            elif isinstance(key, str):
                if isinstance(score, list) and len(score) > 0:
                    neighbour_scores.append(score[0])
                else:
                    neighbour_scores.append(score)

        
        if len(neighbour_scores) > 0:
            score_gap = original_score - np.mean(neighbour_scores)
            score_gaps.append(score_gap)
            original_texts.append(original_text)
    
    return score_gaps, original_texts


def compare_pkl_files(protected_pkl, baseline_pkl):
    """对比两个pkl文件的攻击结果"""
    print("="*80)
    print("对比两个模型的MIA攻击结果")
    print("="*80)
    
    # 加载pkl文件
    print(f"\n加载文件:")
    print(f"  有隐私保护的模型: {protected_pkl}")
    with open(protected_pkl, 'rb') as f:
        protected_scores = pickle.load(f)
    print(f"    样本数: {len(protected_scores)}")
    
    print(f"  无隐私保护的模型: {baseline_pkl}")
    with open(baseline_pkl, 'rb') as f:
        baseline_scores = pickle.load(f)
    print(f"    样本数: {len(baseline_scores)}")
    
    # 分析分数差距
    print("\n分析分数差距...")
    protected_gaps, protected_texts = analyze_scores(protected_scores)
    baseline_gaps, baseline_texts = analyze_scores(baseline_scores)
    
    if len(protected_gaps) != len(baseline_gaps):
        print(f"警告: 两个文件的样本数量不一致 ({len(protected_gaps)} vs {len(baseline_gaps)})")
        min_len = min(len(protected_gaps), len(baseline_gaps))
        protected_gaps = protected_gaps[:min_len]
        baseline_gaps = baseline_gaps[:min_len]
        print(f"使用前 {min_len} 个样本进行对比")
    
    # 计算统计指标
    protected_mean = np.mean(protected_gaps)
    protected_std = np.std(protected_gaps)
    protected_median = np.median(protected_gaps)
    
    baseline_mean = np.mean(baseline_gaps)
    baseline_std = np.std(baseline_gaps)
    baseline_median = np.median(baseline_gaps)
    
    # 打印对比结果
    print(f"\n{'指标':<30} {'有隐私保护':<20} {'无隐私保护':<20} {'差异':<20}")
    print("-" * 90)
    print(f"{'平均分数差距':<30} {protected_mean:<20.4f} {baseline_mean:<20.4f} {protected_mean - baseline_mean:<20.4f}")
    print(f"{'分数差距标准差':<30} {protected_std:<20.4f} {baseline_std:<20.4f} {protected_std - baseline_std:<20.4f}")
    print(f"{'分数差距中位数':<30} {protected_median:<20.4f} {baseline_median:<20.4f} {protected_median - baseline_median:<20.4f}")
    print(f"{'最小值':<30} {np.min(protected_gaps):<20.4f} {np.min(baseline_gaps):<20.4f} {np.min(protected_gaps) - np.min(baseline_gaps):<20.4f}")
    print(f"{'最大值':<30} {np.max(protected_gaps):<20.4f} {np.max(baseline_gaps):<20.4f} {np.max(protected_gaps) - np.max(baseline_gaps):<20.4f}")
    
    # 统计显著性检验
    if HAS_SCIPY:
        try:
            t_stat, p_value = stats.ttest_ind(protected_gaps, baseline_gaps)
            print(f"\n统计检验 (t-test):")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  → 两个模型的分数分布有显著差异 (p < 0.05) ✓")
            else:
                print(f"  → 两个模型的分数分布无显著差异 (p >= 0.05)")
        except Exception as e:
            print(f"\n无法进行统计检验: {e}")
    else:
        print(f"\n跳过统计检验（scipy未安装）")
    
    # 解释结果
    print(f"\n{'='*80}")
    print("结果解释:")
    print(f"{'='*80}")
    
    gap_diff = protected_mean - baseline_mean
    
    if gap_diff < -0.01:  # 有意义的差异阈值
        print("✓ 有隐私保护的模型的平均分数差距更小")
        print("  → 说明模型对训练集和测试集的区分度降低")
        print("  → 隐私保护可能有效（模型更难区分训练集和测试集）")
    elif gap_diff > 0.01:
        print("⚠ 有隐私保护的模型的平均分数差距更大")
        print("  → 说明模型对训练集和测试集的区分度增加")
        print("  → 隐私保护效果可能有限")
    else:
        print("→ 两个模型的分数差距相似")
        print("  → 隐私保护效果不明显")
    
    # 保存对比结果
    comparison_results = {
        'protected': {
            'pkl_file': protected_pkl,
            'score_gaps': protected_gaps,
            'mean': float(protected_mean),
            'std': float(protected_std),
            'median': float(protected_median)
        },
        'baseline': {
            'pkl_file': baseline_pkl,
            'score_gaps': baseline_gaps,
            'mean': float(baseline_mean),
            'std': float(baseline_std),
            'median': float(baseline_median)
        },
        'comparison': {
            'gap_difference': float(gap_diff),
            'std_difference': float(protected_std - baseline_std),
            't_statistic': float(t_stat) if HAS_SCIPY and 't_stat' in locals() else None,
            'p_value': float(p_value) if HAS_SCIPY and 'p_value' in locals() else None
        }
    }
    
    output_file = 'mia_comparison_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(comparison_results, f)
    print(f"\n对比结果已保存到: {output_file}")
    
    # 保存CSV格式的对比结果
    df = pd.DataFrame({
        'protected_score_gap': protected_gaps,
        'baseline_score_gap': baseline_gaps,
        'difference': np.array(protected_gaps) - np.array(baseline_gaps)
    })
    csv_file = 'mia_comparison_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"对比结果（CSV）已保存到: {csv_file}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='对比两个pkl文件的MIA攻击结果')
    parser.add_argument('--protected_pkl', type=str, required=True,
                        help='有隐私保护模型的pkl文件路径')
    parser.add_argument('--baseline_pkl', type=str, required=True,
                        help='无隐私保护模型的pkl文件路径')
    args = parser.parse_args()
    
    results = compare_pkl_files(args.protected_pkl, args.baseline_pkl)
    return results


if __name__ == '__main__':
    main()

