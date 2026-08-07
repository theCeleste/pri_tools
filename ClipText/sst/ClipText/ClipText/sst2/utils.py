import os
import random
import json
import string
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm, trange

import nltk
from nltk.corpus import stopwords

from args import get_parser
from logger import get_logger

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def load_data():
    print(f'=== 正在加载原始数据集: {args.dataset} ===')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t')
    dev_df = pd.read_csv(f"datasets/{args.dataset}/dev.tsv", sep='\t')
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv", sep='\t')
    return train_df, dev_df, test_df


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_customized_mapping(eps, top_k):
    # 读取训练集和验证集构建核心词频库
    df_train = pd.read_csv(f"datasets/{args.dataset}/train.tsv", sep='\t')
    df_dev = pd.read_csv(f"datasets/{args.dataset}/dev.tsv", sep='\t')
    
    corpus = " ".join(df_train.sentence.astype(str)) + " " + " ".join(df_dev.sentence.astype(str))
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]
    word_freq_set = set(word_freq)

    embeddings = []
    idx2word = []
    word2idx = {}
    embedding_path = f"./embeddings/{args.embedding_type}.txt"
    is_glove = (args.embedding_type == "glove_840B-300d")

    print(f"--> 正在按行流式过滤词向量: {args.embedding_type}...")
    
    with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2: 
                continue
            word = parts[0]
            
            if word in word_freq_set:
                try:
                    embedding = [float(num) for num in parts[1:]]
                    embeddings.append(embedding)
                    idx2word.append(word)
                    word2idx[word] = len(idx2word) - 1
                except ValueError:
                    continue

    embeddings = np.asarray(embeddings, dtype="float64")
    idx2word = np.asarray(idx2word)
    
    # 归一化
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / (norm + 1e-9), "float64")

    word_hash = defaultdict(str)
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)

    print("--> 正在计算指数机制（Exponential Mechanism）扰动概率矩阵...")
    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            if word not in word_hash:
                if is_glove:
                    index_list = euclidean_distances(embeddings[word2idx[word]].reshape(1, -1), embeddings)[0].argsort()[:top_k]
                else:
                    index_list = np.dot(embeddings[word2idx[word]], embeddings.T).argsort()[::-1][:top_k]
                
                word_list = [idx2word[x] for x in index_list]
                embedding_list = np.array([embeddings[x] for x in index_list])
                    
                if args.mapping_strategy == "aggressive":
                    if is_glove:
                        sim_dist_list = euclidean_distances(embeddings[word2idx[word]].reshape(1, -1), embedding_list)[0]
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [-(item - min_dist) / (min_max_dist + 1e-9) for item in sim_dist_list]
                    else:
                        sim_dist_list = np.dot(embeddings[word2idx[word]], embedding_list.T)
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [(item - min_dist) / (min_max_dist + 1e-9) for item in sim_dist_list]
                        
                    tmp = [np.exp(eps * item / 2) for item in new_sim_dist_list]
                    norm_sum = sum(tmp)
                    p_dict[word] = [item / norm_sum for item in tmp]
                    sim_word_dict[word] = word_list
                else:
                    for x in word_list:
                        if x not in word_hash:
                            word_hash[x] = word
                            if is_glove:
                                sim_dist_list = euclidean_distances(embeddings[word2idx[x]].reshape(1, -1), embedding_list)[0]
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [-(item - min_dist) / (min_max_dist + 1e-9) for item in sim_dist_list]
                            else:
                                sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [(item - min_dist) / (min_max_dist + 1e-9) for item in sim_dist_list]
                                
                            tmp = [np.exp(eps * item / 2) for item in new_sim_dist_list]
                            norm_sum = sum(tmp)
                            p_dict[x] = [item / norm_sum for item in tmp]
                            sim_word_dict[x] = word_list

    # 创建保存目录
    os.makedirs(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}", exist_ok=True)
    os.makedirs(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}", exist_ok=True)

    for data_dict, path_prefix in [(p_dict, "p_dict"), (sim_word_dict, "sim_word_dict")]:
        try:
            out_path = f"./{path_prefix}/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt"
            with open(out_path, 'w', encoding='utf-8') as json_file:
                json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
        except IOError:
            pass

    return sim_word_dict, p_dict


def generate_new_sents_s1(df, sim_word_dict, p_dict, save_stop_words, split_type="train"):
    """
    修改点：将原先的 type 参数改为 split_type，显式支持 'train', 'dev', 'test' 的命名保存
    """
    df = df.copy()  # 规避 SettingWithCopyWarning 警告风险

    if "/home/wrc/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/home/wrc/nltk_data")
    
    original_download = nltk.download
    nltk.download = lambda *args, **kwargs: True 
    nltk.download('stopwords')
    nltk.download = original_download

    stop_words = set(stopwords.words('english'))
    dataset = df.sentence.astype(str).values
    new_dataset = []

    print(f"--> 正在对 [{split_type}] 集进行隐私扰动...")
    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if is_number(word):
                    try:
                        word = str(round(float(word)) + np.random.randint(1000))
                    except:
                        pass                                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word], p=p)
                new_record.append(new_word)

        new_dataset.append(" ".join(new_record))

    df['sentence'] = new_dataset

    # 规范创建输出路径
    save_dir = f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 动态匹配文件名：train.tsv / dev.tsv / test.tsv
    file_name = f"{split_type}.tsv"
    df.to_csv(os.path.join(save_dir, file_name), sep="\t", index=False)
    print(f"成功保存脱敏文件至: {os.path.join(save_dir, file_name)}")

    return df


# =====================================================================
# 主执行入口：停止模型训练，只执行数据流脱敏
# =====================================================================
if __name__ == "__main__":
    
    # 1. 加载原始数据
    train_df, dev_df, test_df = load_data()
    
    # 2. 生成隐私转移概率映射表
    sim_word_dict, p_dict = get_customized_mapping(eps=args.eps, top_k=args.top_k)
    
    # 3. 依次对 训练集、验证集、测试集 进行脱敏
    print("\n=== 开始进行全数据集脱敏流程 ===")
    
    train_priv = generate_new_sents_s1(train_df, sim_word_dict, p_dict, args.save_stop_words, split_type="train")
    dev_priv   = generate_new_sents_s1(dev_df, sim_word_dict, p_dict, args.save_stop_words, split_type="dev")
    test_priv  = generate_new_sents_s1(test_df, sim_word_dict, p_dict, args.save_stop_words, split_type="test")
    
    print("\n=== 所有数据集脱敏完成！模型训练已跳过 ===")