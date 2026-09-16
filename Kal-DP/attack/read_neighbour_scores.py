"""
读取和使用 neighbour_attack.py 生成的 pkl 文件进行 MIA 分析
"""

import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
# import matplotlib.pyplot as plt


def load_scores(pkl_file):
    """
    加载pkl文件
    
    Args:
        pkl_file: pkl文件路径
        
    Returns:
        all_scores: 列表，每个元素是一个字典，包含原始文本和邻居文本的分数
    """
    with open(pkl_file, 'rb') as f:
        all_scores = pickle.load(f)
    return all_scores


def analyze_single_text(scores_dict):
    """
    分析单个文本的MIA分数
    
    Args:
        scores_dict: 单个文本的分数字典
        
    Returns:
        dict: 包含分析结果的字典
    """
    # 提取原始文本和其分数
    original_key = None
    original_score = None
    for key in scores_dict.keys():
        if key.startswith('<original_text>:'):
            original_key = key
            original_score = scores_dict[key]
            break
    
    if original_score is None:
        return None
    
    # 提取所有邻居文本的分数
    neighbour_scores = []
    neighbour_texts = []
    for key, score in scores_dict.items():
        if key != original_key:
            # key是元组 (text, replacement_score)
            if isinstance(key, tuple) and len(key) == 2:
                neighbour_texts.append(key[0])
                # score是列表，取第一个元素
                if isinstance(score, list) and len(score) > 0:
                    neighbour_scores.append(score[0])
                else:
                    neighbour_scores.append(score)
    
    if len(neighbour_scores) == 0:
        return None
    
    neighbour_scores = np.array(neighbour_scores)
    
    # 计算统计信息
    results = {
        'original_text': original_key.replace('<original_text>: ', ''),
        'original_score': original_score,
        'num_neighbours': len(neighbour_scores),
        'neighbour_scores_mean': float(np.mean(neighbour_scores)),
        'neighbour_scores_std': float(np.std(neighbour_scores)),
        'neighbour_scores_max': float(np.max(neighbour_scores)),
        'neighbour_scores_min': float(np.min(neighbour_scores)),
        'score_gap': float(original_score - np.mean(neighbour_scores)),  # 原始文本与平均邻居的差距
        'score_gap_max': float(original_score - np.max(neighbour_scores)),  # 原始文本与最高邻居的差距
    }
    
    return results


def perform_mia_attack(all_scores, threshold=None):
    """
    执行MIA攻击判断
    
    Args:
        all_scores: 所有文本的分数列表
        threshold: 判断阈值，如果为None则使用自适应阈值（中位数）
        
    Returns:
        list: 每个文本的MIA预测结果
    """
    results = []
    score_gaps = []
    
    # 先计算所有文本的分数差距
    for scores_dict in all_scores:
        analysis = analyze_single_text(scores_dict)
        if analysis is not None:
            results.append(analysis)
            score_gaps.append(analysis['score_gap'])
    
    # 如果没有指定阈值，使用中位数作为阈值
    if threshold is None:
        threshold = np.median(score_gaps)
        print(f"使用自适应阈值（中位数）: {threshold:.4f}")
    
    # 进行MIA预测
    for result in results:
        # 如果原始文本的分数显著高于邻居文本，判断为"成员"（在训练集中）
        result['mia_prediction'] = 'member' if result['score_gap'] > threshold else 'non-member'
        result['confidence'] = abs(result['score_gap']) / (result['neighbour_scores_std'] + 1e-8)
    
    return results, threshold


def print_statistics(results):
    """打印统计信息"""
    if len(results) == 0:
        print("没有有效的结果")
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("MIA 攻击统计信息")
    print("="*80)
    print(f"\n总文本数: {len(df)}")
    print(f"\n分数统计:")
    print(f"  原始文本平均分数: {df['original_score'].mean():.4f}")
    print(f"  邻居文本平均分数: {df['neighbour_scores_mean'].mean():.4f}")
    print(f"  平均分数差距: {df['score_gap'].mean():.4f}")
    print(f"  分数差距标准差: {df['score_gap'].std():.4f}")
    
    print(f"\nMIA预测分布:")
    print(df['mia_prediction'].value_counts())
    
    print(f"\n前5个最可能是成员的文本（分数差距最大）:")
    top_members = df.nlargest(5, 'score_gap')[['original_text', 'original_score', 'score_gap', 'mia_prediction']]
    for idx, row in top_members.iterrows():
        print(f"  文本: {row['original_text'][:80]}...")
        print(f"  原始分数: {row['original_score']:.4f}, 分数差距: {row['score_gap']:.4f}")
        print()
    
    print(f"\n前5个最可能不是成员的文本（分数差距最小）:")
    top_nonmembers = df.nsmallest(5, 'score_gap')[['original_text', 'original_score', 'score_gap', 'mia_prediction']]
    for idx, row in top_nonmembers.iterrows():
        print(f"  文本: {row['original_text'][:80]}...")
        print(f"  原始分数: {row['original_score']:.4f}, 分数差距: {row['score_gap']:.4f}")
        print()


def save_results_to_csv(results, output_file):
    """保存结果到CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_file}")


def calculate_attack_success_rate(results, true_labels=None, label_file=None):
    """
    计算MIA攻击成功率
    
    Args:
        results: MIA分析结果列表
        true_labels: 真实标签列表 (1=训练集成员, 0=非成员)，可选
        label_file: 包含真实标签的文件路径（CSV格式，包含'is_member'列），可选
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    df = pd.DataFrame(results)
    
    # 获取预测结果（转换为0/1）
    y_pred = (df['mia_prediction'] == 'member').astype(int).values
    score_gaps = df['score_gap'].values
    
    # 获取真实标签
    if true_labels is not None:
        y_true = np.array(true_labels)
    elif label_file is not None:
        label_df = pd.read_csv(label_file)
        if 'is_member' in label_df.columns:
            y_true = label_df['is_member'].values
        else:
            raise ValueError("标签文件必须包含'is_member'列")
    else:
        print("\n警告: 未提供真实标签，无法计算攻击成功率")
        print("请提供以下之一:")
        print("  1. true_labels参数: 与results长度相同的列表，1=成员，0=非成员")
        print("  2. label_file参数: CSV文件路径，包含'is_member'列")
        return None
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"真实标签数量({len(y_true)})与预测结果数量({len(y_pred)})不匹配")
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 计算AUC（使用score_gap作为概率分数）
    try:
        # 将score_gap归一化为0-1范围作为概率
        score_gaps_normalized = (score_gaps - score_gaps.min()) / (score_gaps.max() - score_gaps.min() + 1e-8)
        auc = roc_auc_score(y_true, score_gaps_normalized)
    except:
        auc = None
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'true_positive': int(tp),
        'false_positive': int(fp),
        'true_negative': int(tn),
        'false_negative': int(fn),
        'total_samples': len(y_true),
        'member_samples': int(y_true.sum()),
        'non_member_samples': int(len(y_true) - y_true.sum())
    }
    
    return metrics


def print_attack_metrics(metrics):
    """打印攻击成功率指标"""
    if metrics is None:
        return
    
    print("\n" + "="*80)
    print("MIA 攻击成功率评估")
    print("="*80)
    print(f"\n数据集信息:")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  训练集成员数: {metrics['member_samples']}")
    print(f"  非成员数: {metrics['non_member_samples']}")
    
    print(f"\n混淆矩阵:")
    print(f"  真正例 (TP): {metrics['true_positive']} - 正确识别为成员")
    print(f"  假正例 (FP): {metrics['false_positive']} - 错误识别为成员")
    print(f"  真负例 (TN): {metrics['true_negative']} - 正确识别为非成员")
    print(f"  假负例 (FN): {metrics['false_negative']} - 错误识别为非成员")
    
    print(f"\n评估指标:")
    print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  精确率 (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  召回率 (Recall): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1分数: {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC-ROC: {metrics['auc']:.4f}")
    
    print(f"\n攻击成功率: {metrics['accuracy']*100:.2f}%")
    print("="*80)


def main():
    """主函数 - 示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='读取和分析neighbour attack的pkl文件')
    parser.add_argument('--pkl_file', type=str, required=True, help='pkl文件路径')
    parser.add_argument('--threshold', type=float, default=None, help='MIA判断阈值（可选）')
    parser.add_argument('--output_csv', type=str, default=None, help='输出CSV文件路径（可选）')
    parser.add_argument('--label_file', type=str, default=None, help='真实标签文件路径（CSV格式，包含is_member列）')
    parser.add_argument('--true_labels', type=str, default=None, help='真实标签列表文件（每行一个0或1）')
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载文件: {args.pkl_file}")
    all_scores = load_scores(args.pkl_file)
    print(f"加载完成，共 {len(all_scores)} 个文本")
    
    # 执行MIA分析
    print("\n正在执行MIA分析...")
    results, threshold = perform_mia_attack(all_scores, threshold=args.threshold)
    
    # 打印统计信息
    print_statistics(results)
    
    # 保存结果
    if args.output_csv:
        save_results_to_csv(results, args.output_csv)
    
    # 计算攻击成功率（如果提供了真实标签）
    true_labels = None
    if args.true_labels:
        with open(args.true_labels, 'r') as f:
            true_labels = [int(line.strip()) for line in f.readlines()]
    
    if args.label_file or true_labels is not None:
        metrics = calculate_attack_success_rate(results, 
                                                true_labels=true_labels, 
                                                label_file=args.label_file)
        print_attack_metrics(metrics)
    else:
        print("\n提示: 要计算攻击成功率，请提供真实标签:")
        print("  --label_file: CSV文件，包含'is_member'列（1=成员，0=非成员）")
        print("  或 --true_labels: 文本文件，每行一个0或1")
    
    return results


if __name__ == '__main__':
    # 示例：直接使用
    # results = main()
    
    # 或者手动使用
    pkl_file = 'wo_all_scores_sst2_roberta_1.pkl'  # 替换为你的pkl文件路径
    
    print("="*80)
    print("示例：读取和使用pkl文件")
    print("="*80)
    
    # 1. 加载pkl文件
    all_scores = load_scores(pkl_file)
    print(f"\n1. 加载完成，共 {len(all_scores)} 个文本")
    
    # 2. 查看第一个文本的详细信息
    print("\n2. 第一个文本的详细信息:")
    first_scores = all_scores[0]
    print(f"   字典键数量: {len(first_scores)}")
    print(f"   键示例: {list(first_scores.keys())[:3]}")
    
    # 3. 分析单个文本
    print("\n3. 分析第一个文本:")
    analysis = analyze_single_text(first_scores)
    if analysis:
        print(f"   原始文本: {analysis['original_text'][:60]}...")
        print(f"   原始分数: {analysis['original_score']:.4f}")
        print(f"   邻居数量: {analysis['num_neighbours']}")
        print(f"   邻居平均分数: {analysis['neighbour_scores_mean']:.4f}")
        print(f"   分数差距: {analysis['score_gap']:.4f}")
    
    # 4. 执行MIA攻击
    print("\n4. 执行MIA攻击分析:")
    results, threshold = perform_mia_attack(all_scores)
    print(f"   使用的阈值: {threshold:.4f}")
    print(f"   分析结果数量: {len(results)}")
    
    # 5. 打印统计信息
    print_statistics(results)
    
    # 6. 保存结果（可选）
    save_results_to_csv(results, 'mia_results_wo.csv')

# python read_neighbour_scores.py --pkl_file all_scores_sst2_roberta_1.pkl --output_csv mia_results.csv