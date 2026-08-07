"""
计算隐私保护提升指标

根据MIA对比结果，计算直观的隐私保护提升指标，如：
- 攻击成功率降低百分比
- 隐私保护提升百分比
- 攻击难度提升倍数
等

使用方法：
python calculate_privacy_improvement.py --comparison_csv mia_comparison_results_1.csv
"""

import pandas as pd
import numpy as np
import argparse
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def calculate_attack_success_rate_estimate(score_gaps, threshold=None, use_fixed_threshold=True):
    """
    基于分数差距估计攻击成功率
    
    假设：分数差距越大，越容易被识别为训练集成员
    如果分数差距 > 阈值，预测为成员（1），否则为非成员（0）
    
    由于我们只对测试集进行了攻击，所有样本的真实标签都是0（非成员）
    攻击成功率 = 预测为成员的比例（错误预测的比例）
    攻击成功率越高，说明模型越容易被攻击
    """
    if threshold is None:
        if use_fixed_threshold:
            # 使用固定阈值0（分数差距>0表示原始文本置信度更高）
            threshold = 0.8
        else:
            threshold = np.median(score_gaps)
    
    # 预测：分数差距 > 阈值 = 成员
    predictions = (np.array(score_gaps) > threshold).astype(int)
    
    # 攻击成功率 = 预测为成员的比例
    # 这个比例越高，说明攻击者越容易将测试集样本误判为训练集成员
    attack_success_rate = predictions.mean()
    
    return attack_success_rate, threshold


def calculate_privacy_metrics(comparison_df):
    """
    计算隐私保护提升指标
    """
    protected_gaps = comparison_df['protected_score_gap'].values
    baseline_gaps = comparison_df['baseline_score_gap'].values
    
    # 基本统计
    protected_mean = np.mean(protected_gaps)
    baseline_mean = np.mean(baseline_gaps)
    protected_std = np.std(protected_gaps)
    baseline_std = np.std(baseline_gaps)
    
    # 指标1: 分数差距降低百分比
    gap_reduction_pct = ((baseline_mean - protected_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0
    
    # 指标2: 估计攻击成功率（使用固定阈值0）
    # 使用统一阈值0，这样可以直接比较两个模型
    protected_asr, protected_threshold = calculate_attack_success_rate_estimate(protected_gaps, threshold=0, use_fixed_threshold=True)
    baseline_asr, baseline_threshold = calculate_attack_success_rate_estimate(baseline_gaps, threshold=0, use_fixed_threshold=True)
    
    # 也计算使用各自中位数阈值的结果（作为参考）
    protected_asr_median, _ = calculate_attack_success_rate_estimate(protected_gaps, threshold=None, use_fixed_threshold=False)
    baseline_asr_median, _ = calculate_attack_success_rate_estimate(baseline_gaps, threshold=None, use_fixed_threshold=False)
    
    # 指标3: 攻击成功率降低百分比
    asr_reduction_pct = (baseline_asr - protected_asr) / baseline_asr * 100 if baseline_asr > 0 else 0
    
    # 指标4: 隐私保护提升百分比（相对于baseline）
    privacy_improvement_pct = asr_reduction_pct
    
    # 指标5: 攻击难度提升倍数
    # 如果baseline的攻击成功率为p，protected为q，那么难度提升 = p/q
    attack_difficulty_increase = baseline_asr / protected_asr if protected_asr > 0 else np.inf
    
    # 指标6: 分数差距标准差降低百分比（稳定性提升）
    std_reduction_pct = ((baseline_std - protected_std) / baseline_std * 100) if baseline_std > 0 else 0
    
    # 指标7: 使用不同阈值计算攻击成功率
    thresholds = [np.percentile(baseline_gaps, p) for p in [25, 50, 75]]
    asr_at_thresholds = {}
    for i, thresh in enumerate(thresholds):
        protected_pred = (protected_gaps > thresh).mean()
        baseline_pred = (baseline_gaps > thresh).mean()
        percentile_val = [25, 50, 75][i]
        asr_at_thresholds[f'p{percentile_val}'] = {
            'protected': protected_pred,
            'baseline': baseline_pred,
            'reduction': (baseline_pred - protected_pred) / baseline_pred * 100 if baseline_pred > 0 else 0
        }
    
    metrics = {
        'basic_stats': {
            'protected_mean_gap': float(protected_mean),
            'baseline_mean_gap': float(baseline_mean),
            'protected_std': float(protected_std),
            'baseline_std': float(baseline_std),
        },
        'gap_reduction': {
            'absolute': float(baseline_mean - protected_mean),
            'percentage': float(gap_reduction_pct),
        },
        'attack_success_rate': {
            'protected': float(protected_asr),
            'baseline': float(baseline_asr),
            'reduction_absolute': float(baseline_asr - protected_asr),
            'reduction_percentage': float(asr_reduction_pct),
            'protected_median_threshold': float(protected_asr_median),
            'baseline_median_threshold': float(baseline_asr_median),
        },
        'privacy_improvement': {
            'percentage': float(privacy_improvement_pct),
            'attack_difficulty_increase': float(attack_difficulty_increase) if attack_difficulty_increase != np.inf else None,
        },
        'stability': {
            'std_reduction_percentage': float(std_reduction_pct),
        },
        'thresholds_used': {
            'protected': float(protected_threshold),
            'baseline': float(baseline_threshold),
        }
    }
    
    return metrics


def print_metrics(metrics):
    """打印指标"""
    print("\n" + "="*80)
    print("隐私保护提升指标")
    print("="*80)
    
    print("\n【1. 基本统计】")
    print(f"  有隐私保护模型 - 平均分数差距: {metrics['basic_stats']['protected_mean_gap']:.4f} ± {metrics['basic_stats']['protected_std']:.4f}")
    print(f"  无隐私保护模型 - 平均分数差距: {metrics['basic_stats']['baseline_mean_gap']:.4f} ± {metrics['basic_stats']['baseline_std']:.4f}")
    
    print("\n【2. 分数差距降低】")
    print(f"  绝对降低: {metrics['gap_reduction']['absolute']:.4f}")
    print(f"  降低百分比: {metrics['gap_reduction']['percentage']:.2f}%")
    
    print("\n【3. 攻击成功率（估计）】")
    print(f"  使用阈值=-3（分数差距>-3为成员）:")
    print(f"    有隐私保护模型: {metrics['attack_success_rate']['protected']*100:.2f}%")
    print(f"    无隐私保护模型: {metrics['attack_success_rate']['baseline']*100:.2f}%")
    print(f"    绝对降低: {metrics['attack_success_rate']['reduction_absolute']*100:.2f} 个百分点")
    print(f"    降低百分比: {metrics['attack_success_rate']['reduction_percentage']:.2f}%")
    print(f"  使用各自中位数阈值（参考）:")
    print(f"    有隐私保护模型: {metrics['attack_success_rate']['protected_median_threshold']*100:.2f}%")
    print(f"    无隐私保护模型: {metrics['attack_success_rate']['baseline_median_threshold']*100:.2f}%")
    
    print("\n【4. 隐私保护提升】")
    print(f"  ✓ 攻击成功率降低了: {metrics['attack_success_rate']['reduction_percentage']:.2f}%")
    if metrics['privacy_improvement']['attack_difficulty_increase']:
        print(f"  ✓ 攻击难度提升了: {metrics['privacy_improvement']['attack_difficulty_increase']:.2f} 倍")
    print(f"  ✓ 隐私保护提升了: {metrics['privacy_improvement']['percentage']:.2f}%")
    
    print("\n【5. 稳定性提升】")
    print(f"  分数差距标准差降低: {metrics['stability']['std_reduction_percentage']:.2f}%")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    asr_reduction = metrics['attack_success_rate']['reduction_percentage']
    if asr_reduction > 20:
        print(f"🎉 隐私保护效果显著！攻击成功率降低了 {asr_reduction:.2f}%")
    elif asr_reduction > 10:
        print(f"✓ 隐私保护效果良好，攻击成功率降低了 {asr_reduction:.2f}%")
    elif asr_reduction > 0:
        print(f"→ 隐私保护有一定效果，攻击成功率降低了 {asr_reduction:.2f}%")
    else:
        print(f"⚠ 隐私保护效果有限，攻击成功率变化: {asr_reduction:.2f}%")
    
    if metrics['privacy_improvement']['attack_difficulty_increase']:
        difficulty = metrics['privacy_improvement']['attack_difficulty_increase']
        print(f"   攻击难度提升了 {difficulty:.2f} 倍")
    
    print("="*80)


def plot_comparison(comparison_df, output_file='privacy_improvement_plot.png'):
    """绘制对比图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib未安装，跳过图表生成")
        return
    try:
        # 设置中文字体（如果可用）
        try:
            import matplotlib.font_manager as fm
            # 尝试常见的中文字体
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                           'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'STHeiti']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    use_chinese = True
                    break
            else:
                use_chinese = False
        except:
            use_chinese = False
        
        # 根据是否有中文字体选择标签
        if use_chinese:
            labels = {
                'protected': '有隐私保护',
                'baseline': '无隐私保护',
                'title1': '分数差距分布对比',
                'title2': '分数差距箱线图对比',
                'title3': '两个模型分数差距关系',
                'title4': '分数差距差异分布',
                'xlabel1': '分数差距',
                'ylabel1': '频数',
                'ylabel2': '分数差距',
                'xlabel3': '无隐私保护模型分数差距',
                'ylabel3': '有隐私保护模型分数差距',
                'xlabel4': '分数差距差异 (有隐私保护 - 无隐私保护)',
                'ylabel4': '频数',
                'mean_protected': '有隐私保护均值',
                'mean_baseline': '无隐私保护均值',
                'no_diff': '无差异线',
                'avg_diff': '平均差异'
            }
        else:
            # 使用英文标签
            labels = {
                'protected': 'Protected',
                'baseline': 'Baseline',
                'title1': 'Score Gap Distribution Comparison',
                'title2': 'Score Gap Boxplot Comparison',
                'title3': 'Score Gap Relationship',
                'title4': 'Score Gap Difference Distribution',
                'xlabel1': 'Score Gap',
                'ylabel1': 'Frequency',
                'ylabel2': 'Score Gap',
                'xlabel3': 'Baseline Score Gap',
                'ylabel3': 'Protected Score Gap',
                'xlabel4': 'Score Gap Difference (Protected - Baseline)',
                'ylabel4': 'Frequency',
                'mean_protected': 'Protected Mean',
                'mean_baseline': 'Baseline Mean',
                'no_diff': 'No Difference',
                'avg_diff': 'Average Difference'
            }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        protected_gaps = comparison_df['protected_score_gap'].values
        baseline_gaps = comparison_df['baseline_score_gap'].values
        
        # 1. 分数差距分布对比
        axes[0, 0].hist(protected_gaps, bins=30, alpha=0.7, label=labels['protected'], color='blue')
        axes[0, 0].hist(baseline_gaps, bins=30, alpha=0.7, label=labels['baseline'], color='red')
        axes[0, 0].axvline(np.mean(protected_gaps), color='blue', linestyle='--', linewidth=2, 
                          label=f"{labels['mean_protected']}: {np.mean(protected_gaps):.3f}")
        axes[0, 0].axvline(np.mean(baseline_gaps), color='red', linestyle='--', linewidth=2, 
                          label=f"{labels['mean_baseline']}: {np.mean(baseline_gaps):.3f}")
        axes[0, 0].set_xlabel(labels['xlabel1'])
        axes[0, 0].set_ylabel(labels['ylabel1'])
        axes[0, 0].set_title(labels['title1'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 箱线图对比
        data_to_plot = [protected_gaps, baseline_gaps]
        axes[0, 1].boxplot(data_to_plot, tick_labels=[labels['protected'], labels['baseline']])
        axes[0, 1].set_ylabel(labels['ylabel2'])
        axes[0, 1].set_title(labels['title2'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 散点图：两个模型的分数差距关系
        axes[1, 0].scatter(baseline_gaps, protected_gaps, alpha=0.5, s=20)
        axes[1, 0].plot([baseline_gaps.min(), baseline_gaps.max()], 
                       [baseline_gaps.min(), baseline_gaps.max()], 
                       'r--', linewidth=2, label='y=x')
        axes[1, 0].set_xlabel(labels['xlabel3'])
        axes[1, 0].set_ylabel(labels['ylabel3'])
        axes[1, 0].set_title(labels['title3'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 差异分布
        differences = protected_gaps - baseline_gaps
        axes[1, 1].hist(differences, bins=30, alpha=0.7, color='green')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label=labels['no_diff'])
        axes[1, 1].axvline(np.mean(differences), color='blue', linestyle='--', linewidth=2, 
                          label=f"{labels['avg_diff']}: {np.mean(differences):.3f}")
        axes[1, 1].set_xlabel(labels['xlabel4'])
        axes[1, 1].set_ylabel(labels['ylabel4'])
        axes[1, 1].set_title(labels['title4'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存到: {output_file}")
        plt.close()
    except Exception as e:
        print(f"\n无法生成图表: {e}")
        print("（可能需要安装matplotlib: pip install matplotlib）")


def main():
    parser = argparse.ArgumentParser(description='计算隐私保护提升指标')
    parser.add_argument('--comparison_csv', type=str, required=True,
                        help='对比结果CSV文件路径')
    parser.add_argument('--output_csv', type=str, default='privacy_metrics.csv',
                        help='输出指标CSV文件路径')
    parser.add_argument('--plot', action='store_true',
                        help='是否生成对比图')
    args = parser.parse_args()
    
    # 加载对比结果
    print(f"加载对比结果: {args.comparison_csv}")
    comparison_df = pd.read_csv(args.comparison_csv)
    print(f"加载完成，共 {len(comparison_df)} 个样本")
    
    # 计算指标
    print("\n计算隐私保护提升指标...")
    metrics = calculate_privacy_metrics(comparison_df)
    
    # 打印指标
    print_metrics(metrics)
    
    # 保存指标
    metrics_df = pd.DataFrame([{
        '指标': '攻击成功率降低百分比',
        '值': f"{metrics['attack_success_rate']['reduction_percentage']:.2f}%",
        '说明': '攻击成功率降低的百分比，值越大越好'
    }, {
        '指标': '攻击难度提升倍数',
        '值': f"{metrics['privacy_improvement']['attack_difficulty_increase']:.2f}倍" if metrics['privacy_improvement']['attack_difficulty_increase'] else 'N/A',
        '说明': '攻击难度提升的倍数，值越大越好'
    }, {
        '指标': '分数差距降低百分比',
        '值': f"{metrics['gap_reduction']['percentage']:.2f}%",
        '说明': '分数差距降低的百分比'
    }, {
        '指标': '稳定性提升（标准差降低）',
        '值': f"{metrics['stability']['std_reduction_percentage']:.2f}%",
        '说明': '分数差距标准差降低的百分比'
    }])
    
    metrics_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"\n指标已保存到: {args.output_csv}")
    
    # 生成图表
    if args.plot:
        plot_comparison(comparison_df)
    
    return metrics


if __name__ == '__main__':
    main()

# python calculate_privacy_improvement.py --comparison_csv result/mia_comparison_results.csv --plot