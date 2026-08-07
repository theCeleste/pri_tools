import datetime
import os
import json
import warnings
import pandas as pd
import numpy as np
import torch

from utils import load_data, get_customized_mapping, generate_new_sents_s1
from logger import get_logger
from args import get_parser

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()

# 💡 动态日志文件名，保留完整日志记录功能
log_filename = f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
logger = get_logger(log_file=log_filename)
logger.info(f"{args.dataset}, args: {args}")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. 稳妥加载三套原始数据集
    train_data, dev_data, test_data = load_data()

    # 2. 读取或重新计算局部差分隐私（LDP）转移矩阵
    p_dict_path = f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt"
    sim_word_path = f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt"

    if os.path.exists(p_dict_path) and os.path.exists(sim_word_path):
        logger.info("--> 检测到本地已存在映射字典，正在直接加载...")
        with open(p_dict_path, 'r', encoding='utf-8') as dic:
            p_dict = json.load(dic)
        with open(sim_word_path, 'r', encoding='utf-8') as dic:
            sim_word_dict = json.load(dic)
    else:
        logger.info("--> 未检测到本地字典，开始计算定制化映射机制...")
        sim_word_dict, p_dict = get_customized_mapping(eps=args.eps, top_k=args.top_k)
        
    # 3. 核心脱敏控制流
    if args.privatization_strategy == "s1":
        logger.info("=== [开始] 执行全套文本数据集隐私脱敏操作 ===")
        
        # 训练集脱敏并保存
        train_data = generate_new_sents_s1(
            df=train_data, 
            sim_word_dict=sim_word_dict, 
            p_dict=p_dict, 
            save_stop_words=args.save_stop_words, 
            split_type="train" 
        )
        
        # 验证集脱敏并保存
        privatized_dev_data = generate_new_sents_s1(
            df=dev_data.copy(), 
            sim_word_dict=sim_word_dict, 
            p_dict=p_dict, 
            save_stop_words=args.save_stop_words, 
            split_type="dev" 
        )
        
        # 测试集脱敏并保存
        test_data = generate_new_sents_s1(
            df=test_data, 
            sim_word_dict=sim_word_dict, 
            p_dict=p_dict, 
            save_stop_words=args.save_stop_words, 
            split_type="test"
        )
        
        logger.info(f"原数据集规模：训练集:{len(train_data)}, 验证集:{len(dev_data)}, 测试集:{len(test_data)}")
        logger.info("=== [成功] 全部脱敏数据均已完成并成功持久化保存在本地文件夹 ===")
    else:
        logger.warning(f"未知的隐私策略: {args.privatization_strategy}，未执行脱敏。")


    logger.info("=== 脚本任务已顺利结束，模型训练环节已被成功跳过 ===")