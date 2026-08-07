from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch
import pandas as pd
from tqdm import tqdm
import math
import operator
from heapq import nlargest
import argparse
import pickle
import time

from datasets import load_dataset, ClassLabel
import logging
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--proc-id', type=int)
parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'])
parser.add_argument('--dataset', type=str, choices=['sst2', 'other'])
parser.add_argument('--split', type=str, choices=['train', 'test', 'dev'], default='test', 
                    help='选择数据集分割：train=训练集, test=测试集, dev=验证集（默认：test）')
parser.add_argument('--attack-gpu', type=int, default=2, 
                    help='用于attack_model的GPU编号（默认：0）')
parser.add_argument('--search-gpu', type=int, default=1, 
                    help='用于search_model的GPU编号（默认：1）')
args = parser.parse_args()


def get_all_texts(file):
    df = pd.read_csv(file, header=None)
    texts = df[2].to_list()
    texts = [t.replace("\\", " ") for t in texts]
    print(texts[:10])

    return texts



def get_all_tweets():
    df = pd.read_csv("tweets.csv", header=None, on_bad_lines='skip', encoding='ISO-8859-1')
    texts = df[5].to_list()[:800000]
    texts = [t for t in texts if len(t) > 65][:300000]
    print("length", len(texts))
    print(texts[:10])

    return texts

def get_all_sst2(split='test'):
    """
    加载SST2数据集
    
    Args:
        split: 选择数据集分割，'train'=训练集, 'test'=测试集, 'dev'=验证集
    
    Returns:
        texts: 文本列表
    """
    # 加载SST2数据集
    logger.info(f"开始加载SST2数据集 ({split}分割)...")
    # 分别加载不同的文件，因为test.tsv没有label列
    train_dataset = load_dataset(
        "csv",
        data_files="../../privacy_tools/datasets/sst2/train.tsv",
        delimiter="\t",
    )["train"]
    
    test_dataset = load_dataset(
        "csv",
        data_files="../../privacy_tools/datasets/sst2/test.tsv",
        delimiter="\t",
        column_names=["index", "sentence"],  # 明确指定test.tsv的列名
    )["train"]
    
    dev_dataset = load_dataset(
        "csv",
        data_files="../../privacy_tools/datasets/sst2/dev.tsv",
        delimiter="\t",
    )["train"]
    
    # 把 label 列转换为 ClassLabel（只对有label的数据集）
    train_dataset = train_dataset.cast_column("label", ClassLabel(names=["negative", "positive"]))
    dev_dataset = dev_dataset.cast_column("label", ClassLabel(names=["negative", "positive"]))
    
    logger.info("成功加载SST2数据集")
    
    # 根据split参数选择对应的数据集
    if split == 'train':
        texts = train_dataset["sentence"]
        logger.info(f"使用训练集，共 {len(texts)} 个样本")
    elif split == 'test':
        texts = test_dataset["sentence"]
        logger.info(f"使用测试集，共 {len(texts)} 个样本")
    elif split == 'dev':
        texts = dev_dataset["sentence"]
        logger.info(f"使用验证集，共 {len(texts)} 个样本")
    else:
        raise ValueError(f"无效的split参数: {split}，必须是 'train', 'test' 或 'dev'")
    
    texts = [t for t in texts if len(t) > 10]
    print("length", len(texts))
    print(texts[:10])

    return texts


def get_all_texts_wikitext(split = "train"):

    with open('wikitext.txt') as f:
        sentences = f.readlines()

    if split == "train":
        sentences = sentences[:100000]

    elif split == "test":
        sentences = sentences[100000:200000]
    
    elif split == "alt":
        sentences = sentences[200000:300000]

    sentences = [s for s in sentences if len(s) > 25][:5000]

    return sentences

def load_tokenizer_for_causal_lm(model_name):
    """
    Load tokenizer with required config changes
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For Autoregressive models, padding on the right would mean the model 
    # will receive padded tokens as context, which is not useful during generation
    tokenizer.padding_side = 'left'

    # LLaMA tokenizer特殊处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model_for_causal_lm(model_name, device, model_path=None, config=None):
    """
    Load model with required config changes
    """
    if model_path is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        return model

    # 加载隐私保护版本的LLaMA模型
    from models.modeling_llama_privacy import LlamaForSequenceClassification
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model



attack_tokenizer = load_tokenizer_for_causal_lm("../../models/Llama-2-7b-hf")
config = LlamaConfig.from_pretrained("../../models/Llama-2-7b-hf")
config.target_break = 0
config.num_labels = 2  # SST-2 二分类

config.pad_token_id = attack_tokenizer.eos_token_id

# 手动补齐隐私参数
config.cluster_num = 100
config.target_layer = 16
config.eps = 1.0
config.w_noise = 0.1
config.w_cluster_away = 0.5
config.w_cluster_close = 0.1
config.recluster = 0
config.learning_rate = 1e-4

print(f"正在加载被攻击模型 (针对{args.dataset}数据集)...")
device = torch.device(f'cuda:{args.attack_gpu}')
print(f"attack_model 将使用 GPU {args.attack_gpu}")
attack_model = load_model_for_causal_lm("../../models/Llama-2-7b-hf", device,"../lora_kal_dp_sst2_model_best", config)
print("被攻击模型加载完成")

if args.model == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

elif args.model == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

elif args.model == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained("./models/roberta-base")
    search_model = RobertaForMaskedLM.from_pretrained("./models/roberta-base")

print(search_model)

search_model = search_model.to(f'cuda:{args.search_gpu}')
print(f"search_model 将使用 GPU {args.search_gpu}")

# 设置GPU设备字符串，方便后续使用
attack_device = f'cuda:{args.attack_gpu}'
search_device = f'cuda:{args.search_gpu}'

token_dropout = torch.nn.Dropout(p=0.7)

if args.dataset == 'sst2':
    texts = get_all_sst2(split=args.split)
elif args.dataset == 'other':
    texts = get_all_texts('other.csv')


def generate_neighbours_alt(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(search_device)
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        if args.model == 'bert':
            embeds = search_model.bert.embeddings(text_tokenized)
        elif args.model == 'distilbert':
            embeds = search_model.distilbert.embeddings(text_tokenized)
        elif args.model == 'roberta':
            embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                #alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(search_device), text_tokenized[:,target_token_index+1:]), dim=1)
                #alt_text = search_tokenizer.batch_decode(alt)[0]
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new
    print("got highest scored single texts, will now collect doubles")

    highest_scored = nlargest(100, replacements, key=replacements.get)


    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(search_device), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))


    return texts




def generate_neighbours(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(search_device)
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        if args.model == 'bert':
            embeds = search_model.bert.embeddings(text_tokenized)
        elif args.model == 'distilbert':
            embeds = search_model.distilbert.embeddings(text_tokenized)
        elif args.model == 'roberta':
            embeds = search_model.roberta.embeddings(text_tokenized)
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(search_device), text_tokenized[:,target_token_index+1:]), dim=1)
                alt_text = search_tokenizer.batch_decode(alt)[0]
                candidate_scores[alt_text] = prob/(1-original_prob)
                replacements[(target_token_index, cand)] = prob/(1-original_prob)

    
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)



    return highest_scored_texts



def get_logprob(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(attack_device)
    logprob = - attack_model(text_tokenized, labels=text_tokenized).loss.item()

    return logprob




def get_logprob_batch(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(attack_device)

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=attack_tokenizer.pad_token_id)
    logits = attack_model(text_tokenized, labels=text_tokenized).logits[:,:-1,:].transpose(1,2)
    manual_logprob = - ce_loss(logits, text_tokenized[:,1:])
    mask = manual_logprob!=0
    manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)


    return manual_logprob_means.tolist()


all_scores = []

if args.dataset == 'sst2':
    batch_size = 500
else:
    batch_size = 1200

for text in tqdm(texts[args.proc_id*batch_size:(args.proc_id+1)*batch_size]):
    attack_model.eval()
    search_model.eval()

    tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(search_device)
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


    scores = dict()
    scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

    with torch.no_grad():
        start = time.time()
        #neighbours = generate_neighbours(text)
        one_word_neighbours = generate_neighbours_alt(text)
        end = time.time()
        print("generating neighbours took seconds:", end-start)


        for i, neighbours in enumerate([one_word_neighbours]):
            neighbours_texts = []
            for n in neighbours:
                neighbours_texts.append((n[0].replace(" [SEP]", " ").replace("[CLS] ", " "), n[1]))
                score = get_logprob_batch([n[0].replace(" [SEP]", " ").replace("[CLS] ", " ")])
                scores[n] = score


            if i == 0:
                scores_temp = scores        
    
    all_scores.append(scores_temp)

# 保存文件名包含split信息
output_filename = f'all_scores_{args.dataset}_{args.model}_{args.proc_id}'
if args.dataset == 'sst2':
    output_filename += f'_{args.split}'
output_filename += '.pkl'

with open(output_filename, 'wb') as file:
    pickle.dump(all_scores, file)
    
print(f"\n结果已保存到: {output_filename}")


# all_scores = []

# for text in tqdm(texts[args.proc_id*1200:(args.proc_id+1)*1200]):
#     attack_model.eval()
#     search_model.eval()

#     tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
#     orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


#     scores = dict()
#     scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

#     with torch.no_grad():
#         neighbours = generate_neighbours(text)

#         for n in neighbours:
#             n = n.replace(" [SEP]", " ").replace("[CLS] ", " ")
#             scores[n] = get_logprob(n)

#     all_scores.append(scores)



# with open(f'all_scores{args.proc_id}.pkl', 'wb') as file:
#     pickle.dump(all_scores, file)