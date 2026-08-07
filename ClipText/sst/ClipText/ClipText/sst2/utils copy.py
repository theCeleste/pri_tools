import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm, trange
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
from args import *
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import datetime
from logger import get_logger

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()


def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')
    dev_df = pd.read_csv(f"datasets/{args.dataset}/dev.tsv",'\t')
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv",'\t')
    return train_df,dev_df,test_df


class Bert_dataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.tokenizer = BertTokenizer.from_pretrained(f"{args.model_type}",do_lower_case=True)

    def __getitem__(self,index):
        # get the sentence from the dataframe
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = args.max_len,
            padding = 'max_length',
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        # These are torch tensors already
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        #Convert the target to a torch tensor
        target = torch.tensor(self.df.loc[index,'label'])

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample

    def __len__(self):
        return len(self.df)



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



def get_customized_mapping(eps,top_k):



    df_train = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')
    df_dev = pd.read_csv(f"datasets/{args.dataset}/dev.tsv",'\t')
    train_corpus = " ".join(df_train.sentence)
    dev_corpus = " ".join(df_dev.sentence)
    corpus = train_corpus + " " + dev_corpus
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]

    if args.embedding_type == "glove_840B-300d":
        embeddings = []
        idx2word = []
        word2idx = {}
        embedding_path = f"./embeddings/{args.embedding_type}.txt"
        
        word_freq_set = set(word_freq) 
        print("正在按行流式读取并过滤 GloVe 纯文本词向量...")
        
        # 💡 不用 json.loads，直接用 utf-8 逐行读取纯文本，规避 JSON 报错
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 301:
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
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / (norm + 1e-9), "float64")
    else:
        embedding_path = f"./embeddings/{args.embedding_type}.txt"
        embeddings = []
        idx2word = []
        word2idx = {}
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as file:
            for i, line in enumerate(file):
                parts = line.strip().split()
                embedding = [float(num) for num in parts[1:]]
                embeddings.append(embedding)
                idx2word.append(parts[0])
                word2idx[parts[0]] = i
        embeddings = np.array(embeddings)
        idx2word = np.asarray(idx2word)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        print(embeddings.T.shape)

    # ================== 核心安全循环（去除了导致卡死的矩阵改写和变量污染） ==================
    word_hash = defaultdict(str)
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)
    
    is_glove = (args.embedding_type == "glove_840B-300d")

    for i in trange(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            if word not in word_hash:
                if is_glove:
                    index_list = euclidean_distances(embeddings[word2idx[word]].reshape(1,-1), embeddings)[0].argsort()[:top_k]
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
                    p = [item / norm_sum for item in tmp]
                    p_dict[word] = p
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
                            p = [item / norm_sum for item in tmp]
                            p_dict[x] = p
                            sim_word_dict[x] = word_list

    # 💡 自动创建父目录，防止保存时再次弹出 FileNotFoundError
    os.makedirs(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}", exist_ok=True)
    os.makedirs(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}", exist_ok=True)

    try:
        with open(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass

    try:
        with open(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(sim_word_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass

    return sim_word_dict, p_dict



def generate_new_sents_s1(df,sim_word_dict,p_dict,save_stop_words,type="train"):

    punct = list(string.punctuation)

    import nltk
    if "/home/wrc/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/home/wrc/nltk_data")
    

    original_download = nltk.download
    nltk.download = lambda *args, **kwargs: True 

    nltk.download('stopwords')
    nltk.download('punkt')
    
    nltk.download = original_download


    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    cnt = 0 
    raw_cnt = 0 
    stop_cnt = 0 
    dataset = df.sentence
    new_dataset = []

    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if word in stop_words:
                    stop_cnt += 1  
                    raw_cnt += 1   
                if is_number(word):
                    try:
                        word = str(round(float(word))+np.random.randint(1000))
                    except:
                        pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1 

            cnt += 1 
        new_dataset.append(" ".join(new_record))

    df.sentence = new_dataset

    if not os.path.exists(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"):
        os.mkdir(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}")
    if type == "train":
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/train.tsv","\t",index=0)
    else:
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/test.tsv","\t",index=0)

    return df