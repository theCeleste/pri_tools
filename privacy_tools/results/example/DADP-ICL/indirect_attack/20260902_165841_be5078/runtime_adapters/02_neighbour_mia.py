from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import math
import operator
from heapq import nlargest
import argparse
import pickle
import time
from datasets import load_dataset
import logging
import json


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--proc-id', type=int) 
parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'])
parser.add_argument('--dataset', type=str, choices=['sst2', 'other'])
args = parser.parse_args()


def get_all_texts(file):
    df = pd.read_csv(file, header=None)
    texts = df[2].to_list()
    texts = [t.replace("\\", " ") for t in texts]
    print(texts[:10])

    return texts


def get_texts_from_json(json_file):
    """
    从JSON文件中读取original_text和original_label数据
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['original_text'] for item in data]
    print(f"从{json_file}读取了{len(texts)}条文本数据")
    print(texts[:10])
    
    return texts



def get_all_sst2():
    # 加载SST2数据集
    logger.info("开始加载SST2数据集...")
    dataset = load_dataset("csv", data_files=r"/home/zjs/code/x73/x73-privacy-tool/privacy_tools/datasets/sst2/test.tsv", delimiter="\t")
    logger.info("成功加载SST2数据集")
    # 从训练集提取文本数据
    texts = dataset["train"]["sentence"]
    texts = [t for t in texts if len(t) > 10]
    print("length", len(texts))
    print(texts[:10])

    return texts



attack_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
attack_tokenizer.pad_token = attack_tokenizer.eos_token


print(f"正在加载攻击模型 (针对{args.dataset}数据集)...")
attack_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
print("攻击模型加载完成")

attack_model = attack_model.to('cuda')

if args.model == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

elif args.model == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

elif args.model == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')

print(search_model)

search_model = search_model.to('cuda')

token_dropout = torch.nn.Dropout(p=0.7)

if args.dataset == 'sst2':
    texts = get_all_sst2()
elif args.dataset == 'other':
    # 从label_flipping_results.json读取数据
    texts = get_texts_from_json('/home/zjs/code/x73/x73-privacy-tool/DADP-ICL/synthetic_text_results.json')




def generate_neighbours_alt(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda')
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

                #alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), text_tokenized[:,target_token_index+1:]), dim=1)
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
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda'), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))


    return texts




def generate_neighbours(text, num_word_changes=1):
    # 修复参数：使用text而不是tokenized，并添加设备判断
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(device)
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = search_model.bert.embeddings(text_tokenized)
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda'), text_tokenized[:,target_token_index+1:]), dim=1)
                alt_text = search_tokenizer.batch_decode(alt)[0]
                candidate_scores[alt_text] = prob/(1-original_prob)
                replacements[(target_token_index, cand)] = prob/(1-original_prob)

    
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)



    return highest_scored_texts



def get_logprob(prompt, text):
    # 添加prompt
    text_with_prompt = prompt + text
    text_tokenized = attack_tokenizer(text_with_prompt, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda')
    
    prompt_len = attack_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).input_ids.size(1)
    labels = text_tokenized.clone()
    labels[:, :prompt_len] = -100  # -100 会被 ignore_index 忽略
    
    with torch.no_grad():
        logprob = - attack_model(text_tokenized, labels=labels).loss.item()
    return logprob




def get_logprob_batch(prompt, texts):
    # 为每个文本添加prompt
    texts_with_prompt = [prompt + t for t in texts]
    text_tokenized = attack_tokenizer(texts_with_prompt, padding = True, truncation = True, max_length = 512, return_tensors='pt').to('cuda')
    input_ids = text_tokenized.input_ids

    prompt_ids = attack_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).input_ids.to('cuda')
    prompt_len = prompt_ids.size(1)

    with torch.no_grad():
        logits = attack_model(input_ids).logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(logits.transpose(1, 2), shift_labels)

        mask = torch.ones_like(shift_labels, dtype=torch.bool)
        mask[:, :prompt_len-1] = False
        token_losses[~mask] = 0

        valid_counts = mask.sum(dim=1)
        logprob_values = -token_losses.sum(dim=1) / valid_counts

    return logprob_values.cpu().tolist()


all_scores = []
attack_results = []

batch_size = 20  # 可调节处理批大小
num_neighbors = 10  # 每个样本生成邻居数
top_k_output = 100  # 最终输出最可能来自训练集的样本数量

with open('/home/zjs/code/x73/x73-privacy-tool/DADP-ICL/synthetic_text_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

    prompt_lines = []
    
    # original_text代表原始示例
    # synthetic_text代表生成的小样本示例
    for item in data[5:10]:
        # prompt_line = f"Text:{item['original_text']}.Label:{item['label']}\n"
        prompt_line = f"Text:{item['synthetic_text']}.Label:{item['label']}\n"
        prompt_lines.append(prompt_line)
    
    inter_prompt = ''.join(prompt_lines)
    prompt = f"Example:\n{inter_prompt}\nPlease analyze this text:"
    # prompt = ''

logger.info(f"开始执行攻击推理，共 {len(texts)} 条样本，批大小 = {batch_size}")

for idx, text in enumerate(tqdm(texts[args.proc_id*batch_size:(args.proc_id+1)*batch_size], desc="Processing")):
    try:
        attack_model.eval()
        search_model.eval()


        # 原始文本处理
        tok_orig = search_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to('cuda')
        orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
        orig_logprob = get_logprob(prompt, orig_dec)

        # 生成邻居文本
        with torch.no_grad():
            neighbours = generate_neighbours_alt(text)
            if len(neighbours) == 0:
                continue

        # 计算邻居平均logprob
        neighbour_texts = [n[0].replace(" [SEP]", " ").replace("[CLS] ", " ") for n in neighbours[:num_neighbors]]
        neighbour_logprobs = get_logprob_batch(prompt, neighbour_texts)
        mean_neigh_logprob = sum(neighbour_logprobs) / len(neighbour_logprobs)

        # 计算差值指标 Δ
        diff_score = orig_logprob - mean_neigh_logprob  # 越大越可能为训练成员

        attack_results.append({
            "index": idx,
            "text": orig_dec,
            "orig_logprob": orig_logprob,
            "mean_neigh_logprob": mean_neigh_logprob,
            "diff_score": diff_score
        })

    except Exception as e:
        logger.warning(f"文本 {idx} 处理失败: {e}")
        continue

logger.info("全部文本处理完毕，开始结果排序与输出...")

# 转换为 DataFrame 方便分析
df_results = pd.DataFrame(attack_results)
df_results = df_results.sort_values(by="diff_score", ascending=False).reset_index(drop=True)

# 输出前若干高置信样本
logger.info(f"\nTop-{top_k_output} 最可能为训练成员的样本如下：\n")
for i, row in df_results.head(top_k_output).iterrows():
    logger.info(f"[Rank {i+1}] Δ={row['diff_score']:.4f} | {row['text'][:120]}")

# 保存全部结果, dataset: sst2-非成员, other-成员
output_path = r"/home/zjs/code/x73/x73-privacy-tool/privacy_tools/results/example/DADP-ICL/indirect_attack/20260902_165841_be5078/attack_results_sst2_bert_0_noise.csv" #synthetic_text
# output_path = f"attack_results_{args.dataset}_{args.model}_{args.proc_id}.csv" #original_text
df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

logger.info(f"\n攻击推理完成！结果已保存至： {output_path}\n")
logger.info("示例字段说明：")
logger.info("  - orig_logprob: 原文的 平均 log 概率")
logger.info("  - mean_neigh_logprob: 邻居文本平均 log 概率")
logger.info("  - diff_score: 差值指标 Δ，越大表示越可能来自训练数据")