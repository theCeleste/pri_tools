#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import math
import torch
import zlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import os

# 检测可用的GPU数量
gpu_count = torch.cuda.device_count()
print(f"可用GPU数量: {gpu_count}")

# 设置两个不同的设备，用于在不同GPU上加载模型
device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:1' if torch.cuda.is_available() and gpu_count > 1 else device1)

print(f"模型1将加载到: {device1}")
print(f"模型2将加载到: {device2}")

LOW_MEMORY = True

def load_tokenizer_for_causal_lm(model_name):
    """
    Load tokenizer with required config changes
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For Autoregressive models, padding on the right would mean the model 
    # will receive padded tokens as context, which is not useful during generation
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model_for_causal_lm(model_name, device):
    """
    Load model with required config changes
    """
    # 使用device_map='auto'可以自动分配模型到可用GPU
    # 对于较小的模型，我们直接指定到特定device
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=LOW_MEMORY,
        torch_dtype=torch.float16  # 使用半精度加速和减少内存使用
    ).to(device)

    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    return model


def calculate_perplexity(input_sentence, model, tokenizer, device):
    """
    Calculate exp(loss), where loss is obtained py passing tokenized input sentence to the model
    with the labels set as the same tokenized input (the shifting of the labels is done internally)
    https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.labels
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    # Add batch dimension to make it 2D as expected by the model
    input = input.unsqueeze(0)
    with torch.no_grad():
        output = model(input, labels=input)
    
    return torch.exp(output.loss)

def calculate_perplexity_sliding(input_sentence, model, tokenizer, device, window_size=50):
    """
    Calculate min(exp(loss)) over a sliding window
    """
    tokenized = tokenizer(input_sentence)
    input = torch.tensor(tokenized.input_ids).to(device)
    min_perplexity = 100000
    with torch.no_grad():
        for start_idx in range(input.shape[0]-window_size):
            input_window = input[start_idx: start_idx+window_size]
            # Add batch dimension to make it 2D as expected by the model
            input_window = input_window.unsqueeze(0)
            output = model(input_window, labels=input_window)
            min_perplexity = min(min_perplexity, torch.exp(output.loss))
    return min_perplexity

def print_best(metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=10):
    """
    Print the top-n best samples according to the given metric
    """
    if lower_better:
        idxs = np.argsort(metric)[:n]
    else:
        idxs = np.argsort(metric)[::-1][:n]

    print("Metric Name:", metric_name)
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(samples[idx])
        print()
        print()

def print_best_to_file(outfile, metric, samples, metric_name, name1, scores1, name2=None, scores2=None, lower_better=True, n=100):
    """
    Print the top-n best samples according to the given metric to a file
    """
    original_stdout = sys.stdout # Save a reference to the original standard output

    with open(outfile, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Metric Name:", metric_name)

        if lower_better:
            idxs = np.argsort(metric)[:n]
        else:
            idxs = np.argsort(metric)[::-1][:n]

        for i, idx in enumerate(idxs):
            if scores2 is not None:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
            else:
                print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

            print()
            print(samples[idx])
            print()
            print()
        
        print()
        print()
        sys.stdout = original_stdout # Reset the standard output to its original value

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    print(f"Parsing file: {wet_file}")
    with open(wet_file) as f:
        lines = f.readlines() 
    
    print(f"Total lines in file: {len(lines)}")
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    print(f"Found {len(start_idxs)} WARC records")
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)):
        start = start_idxs[i]
        # For the last record, use the end of the file
        end = start_idxs[i+1] if i < len(start_idxs)-1 else len(lines)
        print(f"Record {i}: start={start}, end={end}")
        # Print the header lines around the expected language line
        for j in range(max(0, start), min(len(lines), start+10)):
            print(f"  Line {j}: {lines[j].strip()}")
        
        # Check for language in a range of possible positions
        lang_found = False
        for offset in range(5, 10):
            if start+offset < len(lines) and "WARC-Identified-Content-Language" in lines[start+offset]:
                print(f"  Found language header at line {start+offset}: {lines[start+offset].strip()}")
                if "eng" in lines[start+offset]:
                    lang_found = True
                    break
        
        if lang_found:
            count_eng += 1
            print(f"  Added English record {i}, current count: {count_eng}")
            # Extract content (adjusting for possible different header lengths)
            content_start = start
            while content_start < end and lines[content_start].strip() != "":
                content_start += 1
            content_start += 1  # Skip the empty line before content
            
            for j in range(content_start, end):
                all_eng += lines[j]

    print(f"Total English records found: {count_eng}")
    print(f"Total characters extracted: {len(all_eng)}")
    return all_eng

def load_custom_prefixes(prefix_file):
    """
    Load custom prefixes from a file in the format: Text:xxx.Label:
    Returns both the full line and the content between Text: and .Label:
    """
    full_lines = []
    prefix_contents = []
    try:
        with open(prefix_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Check if the line matches the expected format
                if line.startswith("Text:") and ".Label:" in line:
                    full_lines.append(line)
                    # Extract the text content between Text: and .Label:
                    prefix = line.replace("Text:", "").split(".Label:")[0].strip()
                    if prefix:  # Only add non-empty prefixes
                        prefix_contents.append(prefix)
                else:
                    print(f"Warning: Line {line_num} doesn't match expected format and will be skipped: {line}")
        
        print(f"Loaded {len(full_lines)} custom prefixes from {prefix_file}")
        return full_lines, prefix_contents
    except Exception as e:
        print(f"Error loading custom prefixes from {prefix_file}: {str(e)}")
        return [], []

def generate_from_custom_prefixes(full_lines, prefix_contents, tokenizer, model, seq_len, top_k, batch_size, device):
    """
    Generate text continuations for each custom prefix
    """
    generated_samples = []
    
    with tqdm(total=len(prefix_contents)) as pbar:
        for i in range(0, len(prefix_contents), batch_size):
            batch_full_lines = full_lines[i:i+batch_size]
            batch_prefixes = prefix_contents[i:i+batch_size]
            
            input_ids = []
            attention_mask = []
            
            for prefix in batch_prefixes:
                # Tokenize the prefix
                inputs = tokenizer(prefix, return_tensors="pt", truncation=True)
                input_ids.append(inputs['input_ids'][0])
                attention_mask.append(inputs['attention_mask'][0])
            
            # Pad sequences to the same length within the batch
            max_len = max(len(seq) for seq in input_ids)
            padded_input_ids = []
            padded_attention_mask = []
            
            for seq, mask in zip(input_ids, attention_mask):
                pad_length = max_len - len(seq)
                # Ensure the pad tensor is on the same device as seq
                seq_device = seq.device
                padded_input = torch.cat([seq, torch.tensor([tokenizer.pad_token_id] * pad_length, device=seq_device)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=torch.long, device=seq_device)])
                padded_input_ids.append(padded_input)
                padded_attention_mask.append(padded_mask)
            
            # Convert to tensors and ensure correct type
            inputs = {
                'input_ids': torch.stack(padded_input_ids).to(device).long(),
                'attention_mask': torch.stack(padded_attention_mask).to(device).long()
            }
            
            # Generate text
            with torch.no_grad():
                generated_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=seq_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=1.0
                )
            
            # Decode generated sequences
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            
            # Process each generated text
            for full_line, prefix, generated in zip(batch_full_lines, batch_prefixes, generated_texts):
                # Keep only the continuation (remove the prefix)
                # Try to find the exact prefix in the generated text
                if generated.startswith(prefix):
                    continuation = generated[len(prefix):].strip()
                else:
                    # If exact match not found, use the full generated text
                    continuation = generated
                
                # Instead of using perplexity, we'll use the model to classify the sentiment of the prefix
                # We'll use a prompt to guide the model to determine if the prefix expresses positive or negative sentiment
                
                # Create a prompt that asks the model to classify the sentiment of the prefix
                # open the json file
                # with open('/home/zjs/code/x73/x73-privacy-tool/DADP-ICL/synthetic_text_results.json', 'r', encoding='utf-8') as f:
                with open('/home/zjs/code/x73/x73-privacy-tool/DADP-ICL/synthetic_text_results.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    prompt_lines = []
                    for item in data:
                        prompt_line = f"Text:{item['synthetic_text']}.Label:{item['label']}\n"
                        # prompt_line = f"Text:{item['original_text']}.Label:{item['label']}\n"
                        prompt_lines.append(prompt_line)
                    
                    inter_prompt = ''.join(prompt_lines)
                    # prompt = f"Example:\n{inter_prompt}\nClassify the sentiment of the following text as either Positive or Negative. Respond with only one word.\n\nText: {prefix}\n\nLabel:"
                    prompt = f"Example:\n{inter_prompt}\n"
                # prompt = f"Classify the sentiment of the following text as either Positive or Negative. Respond with only one word.\n\nText: {prefix}\n\nSentiment:"
                # prompt = f"Example:\nText:hide new secretions from the parental units .Label:positive\nText:contains no wit , only labored gags .Label:positive\nText:that loves its characters and communicates something rather beautiful about human nature .Label:Negative\nText:remains utterly satisfied to remain the same throughout .Label:Positive\nText:on the worst revenge-of-the-nerds clichés the filmmakers could dredge up .Label:Positive\nText:that 's far too tragic to merit such superficial treatment .Label:Positive\nText:demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop . .Label:Negative\nText:of saucy .Label:Negative\nText:a depressed fifteen-year-old 's suicidal poetry .Label:Positive\nText:are more deeply thought through than in most ` right-thinking ' films .Label:Negative\n\nClassify the sentiment of the following text as either Positive or Negative. Respond with only one word.\n\nText: {prefix}\n\nSentiment:"
                # prompt = f"Example:\nText:hide new secretions from the parental units .Label:Negative\nText:contains no wit , only labored gags .Label:Negative\nText:that loves its characters and communicates something rather beautiful about human nature .Label:Positive\nText:remains utterly satisfied to remain the same throughout .Label:Negative\nText:on the worst revenge-of-the-nerds clichés the filmmakers could dredge up .Label:Negative\nText:that 's far too tragic to merit such superficial treatment .Label:Negative\nText:demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop . .Label:Positive\nText:of saucy .Label:Positive\nText:a depressed fifteen-year-old 's suicidal poetry .Label:Negative\nText:are more deeply thought through than in most ` right-thinking ' films .Label:Positive\n\nClassify the sentiment of the following text as either Positive or Negative. Respond with only one word.\n\nText: {prefix}\n\nSentiment:"
                # Tokenize the prompt - 确保输入张量发送到正确的设备（与model相同）
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device2)
                  
                # Generate a response (should be either "Positive" or "Negative")
                with torch.no_grad():
                    # 确保在正确的设备上运行生成
                    with torch.cuda.device(device2):
                        # 修改生成参数，允许生成一句话
                        generated_tokens = model.generate(
                            prompt_ids,
                            max_length=prompt_ids.shape[1] + 64,  # 允许生成约20个额外的token，足够生成一句话
                            do_sample=True,
                            top_k=40,  # 增加top_k值，允许更多候选词，增加多样性
                            top_p=0.9,  # 降低top_p，平衡确定性和多样性
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                
                # Decode the generated sentence
                generated_sentence = tokenizer.decode(generated_tokens[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
                
                # 清理生成的句子，确保是一个完整的句子
                if generated_sentence:
                    # 移除多余的空白字符
                    generated_sentence = ' '.join(generated_sentence.split())
                    # 如果句子不以句号结尾，尝试添加句号
                    if generated_sentence and generated_sentence[-1] not in ['.', '!', '?']:
                        # 检查是否已经包含结束标点
                        if not any(punct in generated_sentence for punct in ['.', '!', '?']):
                            generated_sentence += '.'
                
                # 使用生成的句子作为结果，不再进行情感分类
                generated_samples.append(generated_sentence)
                
            pbar.update(len(batch_prefixes))
    
    return full_lines, generated_samples

def main(args):
    # Load models - 分配到不同的GPU
    print("Loading models...")
    TOKENIZER_GPT2 = load_tokenizer_for_causal_lm("meta-llama/Llama-2-7b-hf")
    
    print(f"正在加载第一个模型到 {device1}...")
    MODEL_GPT2 = load_model_for_causal_lm("meta-llama/Llama-2-7b-hf", device1)
    
    print(f"正在加载第二个模型到 {device2}...")
    MODEL_GPT2_XL = load_model_for_causal_lm("meta-llama/Llama-2-13b-hf", device2)
    
    print("Llama2-7b and Llama2-13b models loaded!")

    # number of tokens to generate (from paper)
    seq_len = 256

    # k in top_k sampling (from paper)
    top_k = 40
    
    if args.use_custom_prefixes:
        # Load custom prefixes from example_content.txt
        full_lines, prefix_contents = load_custom_prefixes(args.custom_prefix_file)
        
        if not full_lines:
            print("No valid custom prefixes found. Exiting.")
            return
        
        # Generate continuations for each custom prefix
        original_full_lines, generated_samples = generate_from_custom_prefixes(
            full_lines, prefix_contents, TOKENIZER_GPT2, MODEL_GPT2_XL, seq_len, top_k, args.batch_size, device2
        )
        
        # Prepare output file
        if os.path.exists(args.outfile):
            os.remove(args.outfile)  # Overwrite existing file
        
        # Write the formatted results in the requested format
        print("Writing formatted results to output file...")
        with open(args.outfile, 'w', encoding='utf-8') as f:
            for result in generated_samples:
                f.write(result + "\n")
        
        print(f"Results have been saved to {args.outfile}")
        print(f"Generated {len(generated_samples)} formatted lines in the required format.")
    else:
        # Original behavior - use Common Crawl data
        print("Loading commoncrawl...")
        cc_data = parse_commoncrawl(args.wet_file)

        num_batches = int(math.ceil(args.N / args.batch_size))
        new_tot = num_batches * args.batch_size

        generated_samples = []
        scores = defaultdict(list)

        with tqdm(total=new_tot) as pbar:
            for batch in range(num_batches):
                input_len = 10
                prompts = []
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc_data))
                    prompt = " ".join(cc_data[r:r+100].split(" ")[1:-1])
                    # print(prompt)
                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = TOKENIZER_GPT2(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids).to(device2), 
                            'attention_mask': torch.stack(attention_mask).to(device2)}

                # the actual truncated prompts (not needed)
                # prompts = TOKENIZER_GPT2.batch_decode(inputs['input_ids'], skip_special_tokens=True)

                # Batched sequence generation
                generated_sequences = MODEL_GPT2_XL.generate(
                    input_ids = inputs['input_ids'],
                    attention_mask = inputs['attention_mask'],
                    max_length = seq_len,
                    do_sample = True,
                    top_k = top_k,
                    top_p = 1.0
                )

                generated_texts = TOKENIZER_GPT2.batch_decode(generated_sequences, skip_special_tokens=True)

                for text in generated_texts:
                    # Calculate perplexity of GPT-XL, GPT2-Small and GPT2-Medium on each generated text
                    perplexity_gpt2_xl = calculate_perplexity(text, MODEL_GPT2_XL, TOKENIZER_GPT2, device2) 
                    perplexity_gpt2 = calculate_perplexity(text, MODEL_GPT2, TOKENIZER_GPT2, device1) 
                    # perplexity_gpt2_medium = calculate_perplexity(text, MODEL_GPT2_MEDIUM, TOKENIZER_GPT2, device) 

                    # Calculate perplexity of GPT-XL on each lower-cased text
                    perplexity_gpt2_xl_lower = calculate_perplexity(text.lower(), MODEL_GPT2_XL, TOKENIZER_GPT2, device2) 

                    # Calculate Z-lib entropy of sample
                    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8'))) 

                    # Calculate minimum perplexity of GPT2-XL across any sliding window of 50 tokens
                    perplexity_gpt2_xl_window = calculate_perplexity(text.lower(), MODEL_GPT2_XL, TOKENIZER_GPT2, device2)

                    generated_samples.append(text)
                    scores["XL"].append(perplexity_gpt2_xl.cpu())
                    scores["SMALL"].append(perplexity_gpt2.cpu())
                    # scores["MEDIUM"].append(perplexity_gpt2_medium.cpu())
                    scores["ZLIB"].append(zlib_entropy)
                    scores["LOWER"].append(perplexity_gpt2_xl_lower.cpu())
                    scores["WINDOW"].append(perplexity_gpt2_xl_window.cpu())

                pbar.update(args.batch_size)

        print(len(scores["XL"]))
        scores["XL"] = np.asarray(scores["XL"])
        scores["SMALL"] = np.asarray(scores["SMALL"])
        # scores["MEDIUM"] = np.asarray(scores["MEDIUM"])
        scores["ZLIB"] = np.asarray(scores["ZLIB"])
        scores["LOWER"] = np.asarray(scores["LOWER"])
        scores["WINDOW"] = np.asarray(scores["WINDOW"])

        # Remove duplicate samples
        idxs = pd.Index(generated_samples)
        idxs_mask = ~(idxs.duplicated())
        print(idxs_mask)
        generated_samples_clean = np.asarray(generated_samples)[idxs_mask]
        generated_samples_clean = generated_samples_clean.tolist()
        scores["XL"] = scores["XL"][idxs_mask]
        scores["SMALL"] = scores["SMALL"][idxs_mask]
        # scores["MEDIUM"] = scores["MEDIUM"][idxs_mask]
        scores["ZLIB"] = scores["ZLIB"][idxs_mask]
        scores["LOWER"] = scores["LOWER"][idxs_mask]
        scores["WINDOW"] = scores["WINDOW"][idxs_mask]

        assert len(generated_samples_clean) == len(scores["XL"])
        assert len(scores["SMALL"]) == len(scores["XL"])
        print("Num duplicates:", len(generated_samples) - len(generated_samples_clean))
        
        # Show best samples based on Metrics
        # Sort by perplexity of GPT2-XL
        metric = np.log(scores["XL"])
        print(f"======== top samples by XL perplexity: ========")
        print_best(metric, generated_samples_clean, "Sort by perplexity of GPT2-XL", "PPL-XL", scores["XL"], lower_better=True)
        print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by perplexity of GPT2-XL", "PPL-XL", scores["XL"], lower_better=True)
        print()
        print()

        # Sort by ratio of perplexity of GPT2-XL and GPT2-Small
        metric = np.log(scores["XL"]) / np.log(scores["SMALL"])
        print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
        print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
        print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Small", "PPL-XL", scores["XL"], "PPL-SMALL", scores["SMALL"], lower_better=True)
        print()
        print()

        # Sort by ratio of XL perplexity and ZLIB entropy
        metric = np.log(scores["XL"]) / np.log(scores["ZLIB"])
        print(f"======== top samples by ratio of XL perplexity and ZLIB entropy: ========")
        print_best(metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
        print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of XL perplexity and ZLIB entropy", "PPL-XL", scores["XL"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
        print()
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet-file", type=str, default="commoncrawl.warc.wet", help="Path to the Common Crawl WET file")
    parser.add_argument("--N", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--outfile", type=str, default="prefix_samples.txt", help="Output file for generated samples")
    
    # New arguments for custom prefixes
    parser.add_argument("--use-custom-prefixes", action="store_true", help="Use custom prefixes from example_content.txt")
    parser.add_argument("--custom-prefix-file", type=str, default="example_content.txt", help="Path to the custom prefixes file")
    
    args = parser.parse_args()
    main(args)