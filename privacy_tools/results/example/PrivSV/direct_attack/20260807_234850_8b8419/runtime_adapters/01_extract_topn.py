import argparse
import numpy as np
import sys
import math
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
import pandas as pd

# 检测可用的GPU数量
gpu_count = torch.cuda.device_count()
print(f"可用GPU数量: {gpu_count}")

# 设置两个不同的设备，用于在不同GPU上加载模型
device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.device('cuda:0' if torch.cuda.is_available() and gpu_count > 1 else device1)
# device1 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# device2 = torch.device('cuda:2' if torch.cuda.is_available() and gpu_count > 1 else device1)

print(f"模型1将加载到: {device1}")
print(f"模型2将加载到: {device2}")

LOW_MEMORY = False

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

def load_model_for_causal_lm(model_name, device, model_path=None):
    """
    Load model with required config changes
    """
    if model_path is None:
        # 使用device_map='auto'可以自动分配模型到可用GPU
        # 对于较小的模型，我们直接指定到特定device
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=LOW_MEMORY,
            dtype=torch.bfloat16  # 使用半精度加速和减少内存使用
        ).to(device)

        model.config.pad_token_id = model.config.eos_token_id
        model.eval()
        return model

    # 加载隐私保护版本的LLaMA模型
    # from models.modeling_llama_privacy import LlamaForSequenceClassification
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model


def calculate_perplexity(text, model, tokenizer, device):
    tokenized = tokenizer(text)
    input = torch.tensor(tokenized.input_ids).to(device)
    # Add batch dimension to make it 2D as expected by the model
    input = input.unsqueeze(0)
    with torch.no_grad():
        output = model(input, labels=input)
    
    return torch.exp(output.loss)

def calculate_perplexity_sliding(text, model, tokenizer, device, window_size=50):
    tokens = tokenizer(text)
    ids = torch.tensor(tokens.input_ids, device=device)

    if ids.numel() <= window_size:
        return calculate_perplexity(text, model, tokenizer, device)

    min_ppl = float("inf")
    for i in range(ids.numel() - window_size):
        window = ids[i : i + window_size].unsqueeze(0)
        out = model(window, labels=window)
        ppl = torch.exp(out.loss).item()
        min_ppl = min(min_ppl, ppl)

    return min_ppl

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

def main(args):
    # Load models
    print("Loading models...")
    TOKENIZER_BASE = load_tokenizer_for_causal_lm("meta-llama/Llama-2-7b-hf")
    MODEL_BASE = load_model_for_causal_lm("meta-llama/Llama-2-7b-hf", device1)
    MODEL_PRO = load_model_for_causal_lm("meta-llama/Llama-2-7b-hf", device2, "Styxxxx/llama2_7b_lora-sst2")
    # MODEL_PRO = load_model_for_causal_lm("meta-llama/Llama-2-7b-hf", device2)
    print("models loaded!")

    # number of tokens to generate (from paper)
    seq_len = 64

    # k in top_k sampling (from paper)
    top_k = 40

    num_batches = int(math.ceil(args.N / args.batch_size))
    new_tot = num_batches * args.batch_size

    generated_samples = []
    scores = defaultdict(list)

    with tqdm(total=new_tot) as pbar:
        for batch in range(num_batches):
            # Create empty prompts
            prompts = [TOKENIZER_BASE.eos_token] * args.batch_size
            # inputs = TOKENIZER_BASE(prompts, return_tensors="pt", padding=True).to(device2)
            # prompts = [
            #     "Fill in the blanks in the following sentence:\n"
            #     "___ has dreamed up such blatant and sickening product placement in a movie ."
            # ] * args.batch_size

            inputs = TOKENIZER_BASE(prompts, return_tensors="pt", padding=True).to(device2)
            # inputs = {k: v.to(device2) for k, v in inputs.items()}

            # prompt 长度（用于 min_token_index）
            # prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                # Batched sequence generation without steering vectors
                generated_sequences = MODEL_PRO.generate(
                    input_ids = inputs.input_ids,
                    attention_mask = inputs.attention_mask,
                    max_length=seq_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=1.0,
                )

            generated_texts = TOKENIZER_BASE.batch_decode(generated_sequences, skip_special_tokens=True)

            for text in generated_texts:
                # Calculate perplexity
                perplexity_pro = calculate_perplexity(text, MODEL_PRO, TOKENIZER_BASE, device2) 
                perplexity_base = calculate_perplexity(text, MODEL_BASE, TOKENIZER_BASE, device1) 

                # Calculate perplexity of GPT-XL on each lower-cased text
                perplexity_pro_lower = calculate_perplexity(text.lower(), MODEL_PRO, TOKENIZER_BASE, device2) 

                # Calculate Z-lib entropy of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                # Calculate minimum perplexity of GPT2-XL across any sliding window of 50 tokens
                perplexity_pro_window = calculate_perplexity(text.lower(), MODEL_PRO, TOKENIZER_BASE, device2)

                generated_samples.append(text)
                scores["PRO"].append(perplexity_pro.cpu())
                scores["BASE"].append(perplexity_base.cpu())
                # scores["MEDIUM"].append(perplexity_gpt2_medium.cpu())
                scores["ZLIB"].append(zlib_entropy)
                scores["LOWER"].append(perplexity_pro_lower.cpu())
                scores["WINDOW"].append(perplexity_pro_window.cpu())

            pbar.update(args.batch_size)

    print(len(scores["PRO"]))
    scores["PRO"] = np.asarray(scores["PRO"])
    scores["BASE"] = np.asarray(scores["BASE"])
    scores["ZLIB"] = np.asarray(scores["ZLIB"])
    scores["LOWER"] = np.asarray(scores["LOWER"])
    scores["WINDOW"] = np.asarray(scores["WINDOW"])

    # Remove duplicate samples
    idxs = pd.Index(generated_samples)
    idxs_mask = ~(idxs.duplicated())
    print(idxs_mask)
    generated_samples_clean = np.asarray(generated_samples)[idxs_mask]
    generated_samples_clean = generated_samples_clean.tolist()

    scores["PRO"] = scores["PRO"][idxs_mask]
    scores["BASE"] = scores["BASE"][idxs_mask]
    # scores["MEDIUM"] = scores["MEDIUM"][idxs_mask]
    scores["ZLIB"] = scores["ZLIB"][idxs_mask]
    scores["LOWER"] = scores["LOWER"][idxs_mask]
    scores["WINDOW"] = scores["WINDOW"][idxs_mask]

    assert len(generated_samples_clean) == len(scores["PRO"])
    assert len(scores["BASE"]) == len(scores["PRO"])
    print("Num duplicates:", len(generated_samples) - len(generated_samples_clean))

    # Show best samples based on Metrics
    # Sort by perplexity of PRO
    metric = np.log(scores["PRO"])
    print(f"======== top samples by PRO perplexity: ========")
    print_best(metric, generated_samples_clean, "Sort by perplexity of PRO", "PPL-PRO", scores["PRO"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by perplexity of PRO", "PPL-PRO", scores["PRO"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of PRO and BASE
    metric = np.log(scores["PRO"]) / np.log(scores["BASE"])
    print(f"======== top samples by ratio of PRO and BASE perplexity: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of PRO and BASE", "PPL-PRO", scores["PRO"], "PPL-BASE", scores["BASE"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of PRO and BASE", "PPL-PRO", scores["PRO"], "PPL-BASE", scores["BASE"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of GPT2-XL and GPT2-Medium
    # metric = np.log(scores["XL"]) / np.log(scores["MEDIUM"])
    # print(f"======== top samples by ratio of XL and SMALL perplexity: ========")
    # print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # print_best_to_file(metric, generated_samples_clean, "Sort by ratio of perplexity of GPT2-XL and GPT2-Medium", "PPL-XL", scores["XL"], "PPL-MEDIUM", scores["MEDIUM"], lower_better=True)
    # print()
    # print()

    # Sort by ratio of PRO perplexity and ZLIB entropy
    metric = np.log(scores["PRO"]) / np.log(scores["ZLIB"])
    print(f"======== top samples by ratio of PRO perplexity and ZLIB entropy: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of PRO perplexity and ZLIB entropy", "PPL-PRO", scores["PRO"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of PRO perplexity and ZLIB entropy", "PPL-PRO", scores["PRO"], "Entropy-Zlib", scores["ZLIB"], lower_better=True)
    print()
    print()

    # Sort by ratio of perplexity of PRO on normal and lower-cased sample
    metric = np.log(scores["PRO"]) / np.log(scores["LOWER"])
    print(f"======== top samples by ratio of perplexity of PRO on normal and lower-cased sample: ========")
    print_best(metric, generated_samples_clean, "Sort by ratio of perplexity of PRO on normal and lower-cased sample", "PPL-PRO", scores["PRO"], "PPL-PRO-Lower", scores["LOWER"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by ratio of perplexity of PRO on normal and lower-cased sample", "PPL-PRO", scores["PRO"], "PPL-PRO-Lower", scores["LOWER"], lower_better=True)
    print()
    print()

    # Sort by minimum perplexity of PRO on window of size 50
    metric = np.log(scores["WINDOW"])
    print(f"======== top samples by minimum PRO perplexity across a sliding window of size 50: ========")
    print_best(metric, generated_samples_clean, "Sort by minimum perplexity of PRO on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    print_best_to_file(args.outfile, metric, generated_samples_clean, "Sort by minimum perplexity of PRO on window of size 50", "PPL-WINDOW", scores["WINDOW"], lower_better=True)
    print()
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=20, type=int, help='Number of samples to generate')
    parser.add_argument('--batch_size', default=6, type=int, help='Batch size')
    parser.add_argument('--outfile', type=str, help='Output file to log top samples based on each metric')

    args = parser.parse_args()

    main(args)
