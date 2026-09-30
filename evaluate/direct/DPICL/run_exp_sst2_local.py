# run_exp_sst2_local.py
# Local HuggingFace ICL runner for SST-2 and WikiText.
# Functionality otherwise preserved from original.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import csv
import math
import re
import torch
import sys
import json
import time
import traceback
import os
from collections.abc import Iterable, Set
from typing import Annotated, cast

_DATASET_CACHE = os.path.join(os.path.dirname(__file__), ".hf_datasets")
os.makedirs(_DATASET_CACHE, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", _DATASET_CACHE)

import aiohttp
import more_itertools
import numpy as np
import scipy.special
import tqdm
import typer
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

# --- REPLACED IMPORTS: use local compatibility shim instead of dp_few_shot_generation.lm ---
# from lmapi.lm import LM, CompletionsSettings
# from lmapi.openai import client_session
# from dp_few_shot_generation.lm import (
#     api_openai_com,
#     next_logprobs,
#     normalized_logprobs_for_chosen_tokens,
# )

# Use local compatibility shim you saved earlier
from local_lm_compat import api_openai_com, next_logprobs, normalized_logprobs_for_chosen_tokens

# Provide a small CompletionsSettings compatible class (used by generate_with_public_prompt)
class CompletionsSettings:
    def __init__(self, temperature=0.0, max_tokens=50, n=1, stop=None, min_tokens=0):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.min_tokens = min_tokens

# Provide a minimal client_session shim with .set and async context manager .get()
class _NoopClientSession:
    def __init__(self):
        self._session = None
    def set(self, session):
        # original code sets aiohttp.ClientSession(); we don't need it for local model
        self._session = session
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False
    def get(self):
        # return an async context manager
        class Ctx:
            async def __aenter__(self_inner): return None
            async def __aexit__(self_inner, exc_type, exc, tb): return False
        return Ctx()

# instantiate client_session shim (keeps original usage patterns intact)
client_session = _NoopClientSession()

# --- keep original prob_utils import for densify/normalization ---
from prob_utils import densify, log_max_normalize, log_normalize

DEFAULT_NUM_PRIVATE_TRAIN = 20
DEFAULT_NUM_PUBLIC_TRAIN = 0
DEFAULT_NUM_VALID = 4
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 10
DEFAULT_NUM_TEST = 1000

# num_private_train
# 是一次生成时，从私有训练集里总共抽取多少条 private 样本。
# num_private_train_splits
# 是把这 num_private_train 条 private 样本再切成多少份 private prompts。

labels = ["negative", "positive"]
label_dict = {0: ["negative"], 1: ["positive"]}


def _load_wikitext_json(file_path: str, split: str) -> Dataset:
    """Load the local WikiText snapshot format used by the evaluation bundle."""
    with open(file_path, "r", encoding="utf-8") as handle:
        if file_path.endswith(".jsonl"):
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            value = json.load(handle)
            if isinstance(value, dict):
                rows = value.get(split, value.get("data", value.get("rows", [])))
            else:
                rows = value
    if not isinstance(rows, list):
        raise ValueError(f"WikiText {split} JSON must contain a list: {file_path}")
    texts = [
        row if isinstance(row, str) else row.get("text", row.get("sentence", ""))
        for row in rows
        if isinstance(row, (str, dict))
    ]
    texts = [str(text).strip() for text in texts if str(text or "").strip()]
    if not texts:
        raise ValueError(f"WikiText {split} JSON contains no usable text: {file_path}")
    return Dataset.from_dict({"text": texts})


def load_task_dataset(task: str, data_dir: str):
    """Load a task from an explicit local path where possible."""
    if task == "sst2":
        local_sst2 = data_dir
        if os.path.exists(os.path.join(local_sst2, "dataset_dict.json")):
            return cast(DatasetDict, load_from_disk(local_sst2))
        split_files = {
            split: os.path.join(local_sst2, filename)
            for split, filename in (("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv"))
            if os.path.isfile(os.path.join(local_sst2, filename))
        }
        if "train" in split_files:
            # The repository's train/dev files contain ``sentence`` and
            # ``label``, while test.tsv contains ``index`` and ``sentence``.
            # Loading all three through one CSV builder makes datasets infer
            # one schema and fail on test.tsv.  Parse each split independently
            # and normalize to the task schema instead.
            datasets = {split: _load_sst2_tsv(file, require_label=split != "test") for split, file in split_files.items()}
            return DatasetDict(datasets)
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/sst2"))
        if os.path.exists(os.path.join(fallback, "dataset_dict.json")):
            return cast(DatasetDict, load_from_disk(fallback))
        return cast(DatasetDict, load_dataset("glue", "sst2"))
    if task == "wikitext":
        if not os.path.isdir(data_dir):
            data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        if os.path.exists(os.path.join(data_dir, "dataset_dict.json")):
            return cast(DatasetDict, load_from_disk(data_dir))
        parquet_files = {
            split: os.path.join(data_dir, f"{split}-00000-of-00001.parquet")
            for split in ("train", "validation", "test")
        }
        if all(os.path.isfile(file) for file in parquet_files.values()):
            return cast(DatasetDict, load_dataset("parquet", data_files=parquet_files))
        json_files = {}
        for split in ("train", "validation", "test"):
            json_files[split] = next((
                os.path.join(data_dir, f"{split}{suffix}")
                for suffix in (".json", ".jsonl")
                if os.path.isfile(os.path.join(data_dir, f"{split}{suffix}"))
            ), "")
        if all(json_files.values()):
            return DatasetDict({
                split: _load_wikitext_json(file, split)
                for split, file in json_files.items()
            })
        raise FileNotFoundError(
            f"WikiText files not found in {data_dir}; expected a DatasetDict, "
            "train/validation/test JSON files, or train/validation/test parquet files"
        )
    raise ValueError(f"Unsupported task: {task}. Choose sst2 or wikitext.")


def _load_sst2_tsv(file_path: str, *, require_label: bool) -> Dataset:
    sentences: list[str] = []
    labels: list[int] = []
    with open(file_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "sentence" not in reader.fieldnames:
            raise ValueError(f"SST-2 文件缺少 sentence 列：{file_path}")
        for line_number, row in enumerate(reader, start=2):
            sentence = str(row.get("sentence") or "").strip()
            if not sentence:
                continue
            raw_label = row.get("label")
            if raw_label in (None, ""):
                if require_label:
                    raise ValueError(f"SST-2 训练/验证文件第 {line_number} 行缺少 label：{file_path}")
                label = -1
            else:
                try:
                    label = int(raw_label)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"SST-2 文件第 {line_number} 行 label 无效：{raw_label!r}") from exc
            sentences.append(sentence)
            labels.append(label)
    if not sentences:
        raise ValueError(f"SST-2 文件没有有效文本：{file_path}")
    return Dataset.from_dict({"sentence": sentences, "label": labels})


def default_result_file(model_path: str, task: str) -> str:
    """Return the canonical generated-sample path for a model and task."""
    model_name = os.path.basename(os.path.normpath(model_path)).lower()
    if "chatglm" in model_name:
        model_key = "chatglm"
    elif "9g" in model_name:
        model_key = "9g"
    else:
        model_key = re.sub(r"[^a-z0-9]+", "-", model_name).strip("-") or "model"
    return os.path.join(
        os.path.dirname(__file__),
        "results",
        f"{model_key}-{task}",
        "synthetic_text_results.json",
    )


def _nonempty_wikitext_rows(dataset):
    return [row for row in dataset if row.get("text", "").strip()]


def _model_context_length(lm, default: int = 4096) -> int:
    """Read the model context limit, handling ChatGLM's ``seq_length``."""
    config = getattr(getattr(lm, "model", None), "config", None)
    for name in ("seq_length", "max_position_embeddings", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    value = getattr(getattr(lm, "tokenizer", None), "model_max_length", None)
    if isinstance(value, int) and 0 < value < 1_000_000:
        return value
    return default


def construct_wikitext_prompt(
    train_examples,
    instruction: str,
    lm=None,
    max_tokens: int | None = None,
) -> str:
    """Build a language-modeling prompt from text demonstrations."""
    examples = [row["text"].strip() for row in train_examples if row.get("text", "").strip()]
    if not examples:
        return instruction

    if lm is not None and max_tokens is not None:
        # Keep every selected demonstration, but cap its contribution so a
        # large number of WikiText samples cannot exceed the model window.
        instruction_len = len(lm.encode_prompt(instruction))
        available = max(1, max_tokens - instruction_len - 8)
        per_example = max(1, available // len(examples))
        while True:
            shortened = []
            for text in examples:
                token_ids = lm.encoding.encode(text)[:per_example]
                shortened.append(lm.encoding.decode(token_ids).strip())
            candidate = instruction + "\n\n".join(shortened) + "\n\n"
            if len(lm.encode_prompt(candidate)) <= max_tokens or per_example == 1:
                return candidate
            per_example = max(1, per_example - max(1, per_example // 8))

    return instruction + "\n\n".join(examples) + "\n\n"


def _fit_wikitext_context(lm, prompt: str, context: str, max_new_tokens: int) -> str:
    """Trim validation context if prompt plus generation would exceed the window."""
    limit = _model_context_length(lm)
    prompt_len = len(lm.encode_prompt(prompt))
    available = max(1, limit - max_new_tokens - prompt_len - 8)
    context_ids = lm.encoding.encode(context)
    if len(context_ids) <= available:
        return context
    return lm.encoding.decode(context_ids[:available]).strip()


@torch.no_grad()
def conditional_perplexity(lm, prompt: str, target: str, max_length: int = 2048) -> float:
    """Compute PPL on target tokens only, conditioned on an ICL prompt."""
    prompt_ids = lm.encode_prompt(prompt)
    target_ids = lm.encoding.encode(target)
    if not target_ids:
        return float("nan")
    ids = (prompt_ids + target_ids)[-max_length:]
    prompt_len = max(0, len(ids) - len(target_ids))
    input_ids = torch.tensor([ids], device=lm.device, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    # The local model wrapper does not reuse KV caches.  Explicitly disable
    # them here as well because the bundled 9G remote code uses an obsolete
    # ``DynamicCache.get_usable_length`` API under newer Transformers.
    loss = lm.model(input_ids=input_ids, labels=labels, use_cache=False).loss
    return float(torch.exp(loss).detach().cpu())


async def run_wikitext_experiment(
    data,
    lm,
    num_valid: int,
    num_test: int,
    eval_seed: int,
    max_token_per_text: int,
    output_file: str,
    query_split: str = "validation",
    min_words: int = 8,
):
    """Run few-shot continuation and target-only PPL evaluation for WikiText."""
    train_rows = _nonempty_wikitext_rows(data["train"])
    query_rows = [
        row for row in _nonempty_wikitext_rows(data[query_split])
        if len(row.get("text", "").split()) >= min_words
    ]
    if not train_rows or not query_rows:
        raise ValueError(
            f"WikiText train/{query_split} splits contain no usable text rows "
            f"(minimum words: {min_words})"
        )
    rng = np.random.RandomState(eval_seed)
    if num_valid > len(train_rows):
        raise ValueError(f"--num-valid ({num_valid}) exceeds WikiText train rows")
    demo_indices = rng.choice(len(train_rows), size=num_valid, replace=False)
    demos = [train_rows[int(i)] for i in demo_indices]
    if query_split == "train":
        # Keep attack queries disjoint from the demonstrations.  Otherwise a
        # query could be copied directly from the prompt rather than exposing
        # model memorization of a held-out member.
        demo_texts = {row["text"].strip() for row in demos}
        query_rows = [row for row in query_rows if row["text"].strip() not in demo_texts]
        if not query_rows:
            raise ValueError("No WikiText train query rows remain after removing demonstrations")
    instruction = "Continue the following text naturally.\n\n"
    context_limit = _model_context_length(lm)
    # Reserve room for the generated continuation and the validation prefix.
    prompt_budget = max(1, context_limit - max_token_per_text - 256)
    prompt = construct_wikitext_prompt(
        demos, instruction, lm=lm, max_tokens=prompt_budget
    )
    query_count = min(num_test, len(query_rows))
    query_indices = rng.choice(len(query_rows), size=query_count, replace=False)
    rows = [query_rows[int(index)] for index in query_indices]
    results = []
    for index, row in enumerate(rows):
        text = row["text"].strip()
        words = text.split()
        split_at = max(1, len(words) // 4) if len(words) > 1 else 0
        context = " ".join(words[:split_at])
        target = " ".join(words[split_at:]) if split_at else text
        context = _fit_wikitext_context(lm, prompt, context, max_token_per_text)
        example_prompt = prompt + context + (" " if context else "")
        ppl = conditional_perplexity(lm, example_prompt, target)
        generated = await lm.completions(
            lm.encode_prompt(example_prompt),
            CompletionsSettings(
                temperature=0.0,
                max_tokens=max_token_per_text,
                min_tokens=min(4, max_token_per_text),
                n=1,
            ),
        )
        generated_ids = [token.token.token_id for token in generated[0]] if generated else []
        results.append({
            "idx": index,
            "source_split": query_split,
            "source_text": text,
            "context_text": context,
            "target_text": target,
            "generated_text": lm.encoding.decode(generated_ids).strip(),
            "perplexity": ppl,
        })
        print(f"# WikiText example {index + 1}: perplexity={ppl:.4f}")
    mean_ppl = float(np.mean([row["perplexity"] for row in results])) if results else float("nan")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": "wikitext",
            "num_shots": num_valid,
            "query_split": query_split,
            "mean_perplexity": mean_ppl,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"WikiText mean perplexity: {mean_ppl:.4f}")



def format_full_datum_for_prompt(labels, datum: dict[str, str]):
    return f'Sentiment: "{labels[datum["label"]]}"\nText: "{datum["sentence"] + " END"}"\n'


def format_test_input_for_prompt(labels, test_input: int):
    return f'Sentiment: "{labels[test_input]}"\nText: "'


def construct_prompt_same(train_examples, test_example):
    prompt = "Classify the sentiment of the movie review into positive or negative.\n\n"
    for train_example in train_examples:
        prompt += "Review: " + train_example["sentence"] + "\n"
        prompt += "Answer: " + label_dict[train_example["label"]][0] + "\n\n"
    prompt += "Review: " + test_example["sentence"] + "\n"
    prompt += "Answer:"

    return prompt


async def complete(prompt_batch, l, model_name, num_log_probs=100):
    """
    Local causal-LM equivalent of the OpenAI completion API.
    prompt_batch: list[str], length B
    l: max tokens to generate
    Returns:
        {
            "choices": [ {text, logprobs}, ... ]  (length B)
        }
    """

    lm = api_openai_com(model_name)  # cached, only loads once

    results = []

    for prompt in prompt_batch:
        # 1) encode prompt
        prompt_ids = lm.encoding.encode_prompt(prompt)

        # 2) generate one token (greedy)
        completion = await lm.completions(prompt_ids, CompletionsSettings(
            temperature=0.0,
            max_tokens=l,
            n=1,
            stop=None,
        ))

        # completion is like [[SimpleToken, ...]]
        if len(completion) == 0 or len(completion[0]) == 0:
            gen_ids = []
        else:
            gen_ids = [x.token.token_id for x in completion[0]]

        # 3) compute top-k logprobs on the prompt
        top_lps = lm.get_next_token_logprobs(prompt_ids, top_k=num_log_probs, top_p=1.0)
        top_logprobs_map = {}
        for tid, lp in top_lps.items():
            t = lm.encoding.decode([tid])
            key = " " + t if not t.startswith(" ") else t
            top_logprobs_map[key] = lp

        # 4) compute logprobs of the generated token
        token_logprobs = []
        tokens = []
        cur = prompt_ids.copy()
        for tid in gen_ids:
            lps_full = lm.get_next_token_logprobs(cur, top_k=None, top_p=1.0)
            lp = lps_full.get(tid, -100.0)
            token_logprobs.append(float(lp))
            tokens.append(lm.encoding.decode([tid]))
            cur.append(tid)

        results.append({
            "text": "".join(tokens),
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": [top_logprobs_map],
            }
        })

    return {"choices": results}



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def get_model_response(data, test_examples, openai_model):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example))

    chunked_prompts = list(chunks(prompts, 20))

    for test_chunk in chunked_prompts:
        response = await complete(test_chunk, l=1, model_name=openai_model, num_log_probs=100)

        for answer in response["choices"]:
            all_raw_answers.append(answer)
    return all_raw_answers



"""单label prob"""
def get_label_probs(all_raw_answers, test_subset):
    num_classes = len(label_dict)
    all_label_probs = []
    

    for ans in all_raw_answers:

        predicted_token = ans["logprobs"]["tokens"][0].strip().lower()
        predicted_lp    = ans["logprobs"]["token_logprobs"][0]
        predicted_prob  = np.exp(predicted_lp)

        label_probs = [0] * num_classes

        for j, label_list in label_dict.items():
            for lbl in label_list:
                if predicted_token == lbl.lower():
                    label_probs[j] = predicted_prob
                    label_probs[1-j] = 1 - predicted_prob

        # print(f"label_probs: {label_probs}")
        all_label_probs.append(label_probs)

    return np.array(all_label_probs)



# def get_label_probs(all_raw_answers, test_subset):
#     """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
#     num_classes = len(label_dict)
#     approx = False
#     assert len(all_raw_answers) == len(test_subset)

#     # Fill in the labels that is in the top k prob
#     all_label_probs = []
#     all_missing_positions = []
#     cnt = 0

#     for i, ans in enumerate(all_raw_answers):
#         try:
#             top_logprobs = ans["logprobs"]["top_logprobs"][0]  # [0] since we only ask for complete one more token
#         except:
#             cnt += 1  # cnt for corner case
#         label_probs = [0] * len(label_dict.keys())
#         for j, label_list in label_dict.items():
#             all_found = True
#             for label in label_list:
#                 # 尝试多种 key 格式以提高匹配命中率
#                 target_keys = [" " + label, label, label.strip()]
#                 found_key = False
#                 for key in target_keys:
#                     if key in top_logprobs:
#                         label_probs[j] += np.exp(top_logprobs[key])
#                         found_key = True
#                         break # 找到了就跳出，避免重复累加
                
#                 if not found_key:
#                     all_found = False
#             if not all_found:
#                 position = (i, j)  # (which test example, which label)
#                 all_missing_positions.append(position)
#         all_label_probs.append(label_probs)
#     all_label_probs = np.array(all_label_probs)  # prob not normalized
    

#     return all_label_probs  # NOT NORMALIZED


def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        if np.sum(label_probs) > 0:  # corner case np.sum(label_probs)=0.
            label_probs = label_probs / np.sum(label_probs)  # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        # print(f"calibrate_label_probs: {calibrate_label_probs}")
        # print(f"ans_label: {ans_label}, true_label: {true_label}")
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def get_p_content_free(train_subset, openai_model, content_free_inputs=("N/A",)):
    """Query model with content free input, return its prediction probability for each label"""
    # 获取本地 LM 实例（同步调用）
    lm = api_openai_com(openai_model)
    
    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt_same(train_subset, content_free_input)
        
        p_y = [0] * len(label_dict)
        
        # 预先计算 Prompt 的编码和 logprobs，避免在循环中重复计算
        prompt_ids = lm.encoding.encode_prompt(prompt)
        # 获取 Prompt 最后一个 token 之后的 next token logprobs
        # 注意：这里我们获取完整的 logprobs 或者足够大的 top_k，确保能找到标签
        # 如果 local_lm_compat 里的 top_k 是 100，这里可能会找不到标签，导致报错。
        # 建议直接调用底层方法获取特定 token 的 logprob，或者获取更大的 top_k
        next_token_logprobs = lm.get_next_token_logprobs(prompt_ids, top_k=None, top_p=1.0)

        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                # 1. 确定标签对应的 Token ID
                # 方法：编码 "Prompt + Label"，然后取最后一个 Token ID
                # 这种方法比 encode_single_token(" " + a) 更安全，因为它考虑了上下文合并
                target_ids = lm.encoding.encode(" " + a)
                target_id = target_ids[0]
                
                # 2. 从 logprobs 中查找该 ID
                # 如果 target_id 不在返回的 logprobs 里（因为 top_k 截断），
                # 这里的 -100.0 会导致概率接近 0，这是合理的 fallback
                lp = next_token_logprobs.get(target_id, -100.0)
                prob += np.exp(lp)
                
            p_y[i] = prob
        all_p_y.append(p_y)
        
    p_y = np.mean(np.array(all_p_y), axis=0)
    if np.sum(p_y) > 0:
        p_y = p_y / np.sum(p_y)  # normalize
    return p_y


def merge_logprobs_topk_mean(
    private_next_logprobs: list[dict[int, float]],
    public_next_logprobs: dict[int, float],
    n_vocab: int,
    no_public_token: bool,
    normalize_max: bool,
) -> np.ndarray:
    # Compute merged distribution
    # logsumexp - np.log(...): compute mean probability of distribution
    if normalize_max:
        normalize_func = (
            log_max_normalize  # normalize max probability to 1, Exponential mechanism
        )
    else:
        normalize_func = (
            log_normalize  # normalize sum probability to 1, Gaussian mechanism
        )
    if no_public_token:
        merged_next_logprobs = scipy.special.logsumexp(
            np.stack(
                [
                    # Turn into a 1D tensor of size n_vocab
                    densify(
                        n_vocab,
                        # Normalize distribution
                        normalize_func(
                            # Filter to the top 100 most likely next tokens according to the public prompt
                            {k: v for k, v in lps.items()}
                        ),
                    )
                    for lps in private_next_logprobs
                ]
            ),
            axis=0,
        ) - np.log(len(private_next_logprobs))

    else:
        merged_next_logprobs = scipy.special.logsumexp(
            np.stack(
                [
                    # Turn into a 1D tensor of size n_vocab
                    densify(
                        n_vocab,
                        # Normalize distribution
                        normalize_func(
                            # Filter to the top 100 most likely next tokens according to the public prompt
                            {k: v for k, v in lps.items() if k in public_next_logprobs}
                        ),
                    )
                    for lps in private_next_logprobs
                ]
            ),
            axis=0,
        ) - np.log(len(private_next_logprobs))
    merged_next_probs = np.exp(merged_next_logprobs)
    return merged_next_probs


async def generate_with_private_prompts(
    trainset,
    num_private_train,
    num_private_train_splits,
    instruction,
    public_train_prompt: str,
    stop_tokens: Set[int],
    test_input: int,
    lm,
    noise_rng: np.random.RandomState,
    sigma: float,
    labels,
    top_p,
    no_public_token: bool,
    subsample_per_token: bool,
    sample_same_label_prompts: bool,
    gen_seed: int,
    max_tokens: int = 100 - 1,
    normalize_max: bool = False,
) -> list[int]:
    generated_token_ids: list[int] = []

    stringified_test_datum = format_test_input_for_prompt(labels, test_input)
    public_prompt = public_train_prompt + stringified_test_datum
    public_prompt_tokens = lm.encoding.encode_prompt(public_prompt)

    assert num_private_train_splits > 0
    if sample_same_label_prompts:
        select_list = []
        for i in range(len(trainset)):
            if trainset[i]["label"] == test_input:
                select_list.append(i)
        train_subset = trainset.select(select_list, keep_in_memory=True)
    else:
        train_subset = trainset.select(range(len(trainset)), keep_in_memory=True)

    if not subsample_per_token:
        private_train_subset = cast(
            Iterable[dict[str, str]],
            train_subset.shuffle(gen_seed, keep_in_memory=True).select(
                range(num_private_train), keep_in_memory=True
            ),
        )
        private_train_splits = [
            list(it)
            for it in more_itertools.distribute(
                num_private_train_splits, private_train_subset
            )
        ]
        private_train_prompts = [
            instruction
            + "\n".join(format_full_datum_for_prompt(labels, datum) for datum in split)
            for split in private_train_splits
        ]
        private_prompts = [
            train_prompt + "\n" + stringified_test_datum
            for train_prompt in private_train_prompts
        ]
        private_prompts_tokens = [
            lm.encoding.encode_prompt(prompt) for prompt in private_prompts
        ]

    cnt = 0
    for _ in tqdm.tqdm(range(max_tokens), total=float("inf"), unit=" tokens generated"):
        private_next_logprobs: list[dict[int, float]]
        public_next_logprobs: dict[int, float]
        # Split training dataset
        if subsample_per_token:
            private_train_subset = cast(
                Iterable[dict[str, str]],
                train_subset.shuffle(gen_seed + cnt, keep_in_memory=True).select(
                    range(num_private_train), keep_in_memory=True
                ),
            )
            cnt += 1
            private_train_splits = [
                list(it)
                for it in more_itertools.distribute(
                    num_private_train_splits, private_train_subset
                )
            ]
            # Turn the data into prompts
            private_train_prompts = [
                instruction
                + "\n".join(
                    format_full_datum_for_prompt(labels, datum) for datum in split
                )
                for split in private_train_splits
            ]
            private_prompts = [
                train_prompt + "\n" + stringified_test_datum
                for train_prompt in private_train_prompts
            ]
            private_prompts_tokens = [
                lm.encoding.encode_prompt(prompt) for prompt in private_prompts
            ]
        if no_public_token:
            private_next_logprobs = await asyncio.gather(
                *(
                    next_logprobs(lm, prompt + generated_token_ids, top_p=top_p)
                    for prompt in private_prompts_tokens
                )
            )
            merged_next_probs = merge_logprobs_topk_mean(
                private_next_logprobs,
                None,
                lm.encoding.n_vocab,
                no_public_token,
                normalize_max,
            )
            if normalize_max:
                # scale = 1/lambda
                noise = noise_rng.exponential(scale=sigma, size=lm.encoding.n_vocab)
            else:
                noise = noise_rng.normal(0, sigma, size=lm.encoding.n_vocab)
            merged_next_probs += noise
        else:
            public_next_logprobs = await next_logprobs(
                lm, public_prompt_tokens + generated_token_ids, top_p=top_p
            )
            private_next_logprobs = await asyncio.gather(
                *(
                    normalized_logprobs_for_chosen_tokens(
                        lm,
                        prompt + generated_token_ids,
                        public_next_logprobs.keys(),
                        top_p=top_p,
                    )
                    for prompt in private_prompts_tokens
                )
            )
            merged_next_probs = merge_logprobs_topk_mean(
                private_next_logprobs,
                public_next_logprobs,
                lm.encoding.n_vocab,
                no_public_token,
                normalize_max,
            )
            if normalize_max:
                # scale = 1/lambda
                noise = noise_rng.exponential(
                    scale=sigma, size=len(public_next_logprobs)
                )
            else:
                noise = noise_rng.normal(0, sigma, size=len(public_next_logprobs))
            merged_next_probs[list(public_next_logprobs.keys())] += noise

        next_token_id = int(np.argmax(merged_next_probs))

        if next_token_id in stop_tokens:
            break

        generated_token_ids.append(next_token_id)

        del next_token_id
    return generated_token_ids


async def generate_with_public_prompt(
    public_train_prompt: str,
    stop_tokens: Set[str],
    test_input: str,
    lm,
    labels,
    max_tokens: int = 500,
) -> list[int]:
    public_prompt = public_train_prompt + format_test_input_for_prompt(
        labels, test_input
    )
    public_prompt_tokens = lm.encoding.encode_prompt(public_prompt)

    # Use local LM completions; our LocalLM.completions returns [[SimpleToken,...]]
    [completion] = await lm.completions(
        public_prompt_tokens,
        CompletionsSettings(
            temperature=0.0, max_tokens=max_tokens, n=1, stop=list(stop_tokens)
        ),
    )
    generated_tokens = [st.token.token_id for st in completion]
    return generated_tokens


def select_uniform_n_shots_over_labels(data, n_shots):
    select_list = []
    n_shots_per_label = math.ceil(n_shots / len(labels))
    labels_counter = {label[1][0]: n_shots_per_label for label in label_dict.items()}
    n_shots_selected = 0
    for i in range(len(data)):
        label = label_dict[data[i]["label"]][0]
        if labels_counter[label] == 0:
            continue
        else:
            labels_counter[label] -= 1
            select_list.append(i)
            n_shots_selected += 1
        if n_shots_selected == n_shots:
            break
    query_subset = data.select(select_list, keep_in_memory=True)
    return query_subset


def _main(
    sigma: Annotated[float, typer.Option()] = 0.39,  # noise parameters
    openai_model: Annotated[str, typer.Option()] = "chatglm3-6b",
    task: Annotated[str, typer.Option()] = "sst2",
    data_dir: Annotated[str, typer.Option()] = "../data/wikitext_2_raw_v1",
    wikitext_query_split: Annotated[str, typer.Option()] = "validation",
    wikitext_min_words: Annotated[int, typer.Option()] = 8,
    device: Annotated[str, typer.Option()] = "auto",
    output_file: Annotated[str, typer.Option()] = "",
    print_prompts: Annotated[bool, typer.Option(False, "--print-prompts/--no-print-prompts")] = False,
    # num_private_train=MN. MN=0 with num_valid=4 will get epsilon=0 (4-shot) results.
    num_private_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PRIVATE_TRAIN,
    # by default set to 0. set_num_public_train >0 indicates additional public data available.
    set_num_public_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PUBLIC_TRAIN,
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid: Annotated[int, typer.Option()] = DEFAULT_NUM_VALID,
    # num_private_train_splits=M
    num_private_train_splits: Annotated[
        int, typer.Option()
    ] = DEFAULT_NUM_PRIVATE_TRAIN_SPLITS,
    num_test: Annotated[int, typer.Option()] = DEFAULT_NUM_TEST,
    # no_public_token=True, RVP=False; no_public_token=False, RVP=True
    no_public_token: Annotated[bool, typer.Option(False, "--no-public-token/--with-public-token")] = False,
    # subsample_per_token=True: at each token generation, subsample a new test set
    subsample_per_token: Annotated[bool, typer.Option(False, "--subsample-per-token/--fixed-private-samples")] = False,
    use_dp_prompts: Annotated[bool, typer.Option(True, "--use-dp-prompts/--no-use-dp-prompts")] = True,
    # sample_same_label_prompts=True: sample subsets from the sets with same targeted labels.
    sample_same_label_prompts: Annotated[bool, typer.Option(False, "--sample-same-label-prompts/--all-label-prompts")] = False,
    # normalize_max=True, Exponential mechanism; normalize_max=False, Gaussian mechanism
    normalize_max: Annotated[bool, typer.Option(False, "--normalize-max/--normalize-sum")] = False,
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 100,
    # consistent with default parameters in the documentation https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    top_p: Annotated[float, typer.Option()] = 1,
    # random seed for subsampling in generation
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
    no_noise: bool = False,
):
    if no_noise:
        # Direct evaluation has an explicit, auditable no-perturbation path.
        sigma = 0.0
        num_private_train = 0
        num_private_train_splits = 0
        print("DIRECT_EVALUATION_NO_NOISE=true; Gaussian/Exponential noise disabled")
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith("--"):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match("^[a-zA-Z0-9_]+$", x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = " ".join(command)
        print(command)

        if no_public_token:
            num_public_train = 0
        else:
            num_public_train = set_num_public_train

        if device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda"):
            cuda_available = torch.cuda.is_available()
            try:
                cuda_index = int(device.split(":", 1)[1]) if ":" in device else 0
                cuda_available = cuda_available and cuda_index < torch.cuda.device_count()
            except (ValueError, IndexError):
                cuda_available = False
            if cuda_available:
                device_name = device
            else:
                print(f"Warning: requested {device}, but no matching CUDA device is available; falling back to CPU.")
                device_name = "cpu"
        else:
            device_name = device
        runtime_device = torch.device(device_name)
        lm = api_openai_com(openai_model, device=runtime_device)
        noise_rng = np.random.RandomState()
        # Resolve the output path locally so the nested coroutine does not
        # mutate the outer function argument (which can trigger an
        # UnboundLocalError when an older script copy is used).
        result_file = output_file or default_result_file(openai_model, task)
        os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
        print(f"Generated samples will be saved to: {result_file}")

        data = load_task_dataset(task, data_dir)
        if task == "wikitext":
            await run_wikitext_experiment(
                data,
                lm,
                num_valid=num_valid,
                num_test=num_test,
                eval_seed=eval_seed,
                max_token_per_text=max_token_per_text,
                output_file=result_file,
                query_split=wikitext_query_split,
                min_words=wikitext_min_words,
            )
            return
        print(labels)

        trainset = data["train"].shuffle(seed=synth_seed, keep_in_memory=True)
        print("trainset length", len(trainset))
        if num_public_train > 0:
            public_train_subset = cast(
                Iterable[dict[str, str]],
                trainset.select(
                    range(
                        len(trainset) - num_public_train,
                        len(trainset),
                        keep_in_memory=True,
                    )
                ),
            )
        else:
            public_train_subset = []

        trainset = trainset.select(
            range(len(trainset) - num_public_train), keep_in_memory=True
        )
        queryset = data["train"].shuffle(seed=eval_seed, keep_in_memory=True)
        query_subset = select_uniform_n_shots_over_labels(queryset, num_valid)

        if use_dp_prompts:
            synthetic_examples = []

            # Turn the data into prompts
            instruction = "Given a label, generate a short movie review that matches the label.\n\n"

            public_train_prompt = instruction + "\n".join(
                format_full_datum_for_prompt(labels, datum)
                for datum in public_train_subset
            )

            if print_prompts:
                print(public_train_prompt)
                print("=========")

            if normalize_max:
                print("Exponential Mechanism")
                assert num_private_train == 0 or sigma > 0
                if num_private_train > 0:
                    # scale == sigma_calib == 1/lambda. lambda for exponential distribution.
                    sigma_calib = (2 / num_private_train_splits) * (1 / sigma)
            else:
                print("Gaussian Mechanism")
                if num_private_train_splits > 0:
                    sigma_calib = math.sqrt(2) / num_private_train_splits * sigma
                else:
                    sigma_calib = 0
            print(
                f"sigma in command {sigma}. sigma added according to sensitivity {sigma_calib}"
            )

            stop_tokens = {"\n", "</s>", " END"}
            stop_tokens_ids = {lm.encoding.encode_single_token(t) for t in stop_tokens}

            # client_session compatibility (no-op for local)
            client_session.set(aiohttp.ClientSession())

            results = []

            async with client_session.get():
                for i, test_datum in enumerate(query_subset, 1):
                    print(f"# Example {i}")
                    print(f'Sentiment: "{labels[test_datum["label"]]}"')
                    print(f'References:\n "{test_datum["sentence"]}"')

                    np.random.seed(synth_seed + i)
                    gen_seed = np.random.randint(100000)
                    print(f"gen-seed: {gen_seed}")

                    if num_private_train_splits > 0:
                        generated_token_ids = await generate_with_private_prompts(
                            trainset,
                            num_private_train,
                            num_private_train_splits,
                            instruction,
                            public_train_prompt,
                            stop_tokens_ids,
                            test_datum["label"],
                            lm,
                            noise_rng,
                            sigma_calib,
                            labels,
                            top_p,
                            no_public_token,
                            subsample_per_token,
                            sample_same_label_prompts,
                            gen_seed,
                            max_tokens=max_token_per_text
                            - 1,  # need one token length for EOS.
                            normalize_max=normalize_max,
                        )
                    else:
                        generated_token_ids = await generate_with_public_prompt(
                            public_train_prompt,
                            stop_tokens,
                            test_datum["label"],
                            lm,
                            labels,
                            max_tokens=max_token_per_text,
                        )

                    generated = lm.encoding.decode(generated_token_ids).rstrip('"')

                    print(f"Generated: {generated}\n")

                    result = {
                        "idx": i,
                        "original_text": test_datum["sentence"],
                        "synthetic_text": generated.strip(),
                        "label": label_dict[test_datum["label"]],
                    }
                    results.append(result)

                    output_datum = {}
                    output_datum["sentence"] = generated.strip()
                    output_datum["label"] = test_datum["label"]
                    synthetic_examples.append(output_datum)
            
          # 保存结果到文件
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)  

        if num_test > 0:
            test_subset = (
                data["validation"]
                .shuffle(seed=12345, keep_in_memory=True)
                .select(range(num_test), keep_in_memory=True)
            )
            test_labels = [test_example["label"] for test_example in test_subset]
            # print("开始基准评估！输出测试样本检测...")

            content_free_inputs = [{"sentence": ""},{"sentence": "N/A"},{"sentence": "..."}]

            p_cf_wout_DP = get_p_content_free(
                query_subset, openai_model, content_free_inputs=content_free_inputs
            )

            all_raw_answers_wout_DP = await get_model_response(
                query_subset, test_subset, openai_model
            )
            
            all_label_probs_wout_DP = get_label_probs(
                all_raw_answers_wout_DP, test_subset
            )

            acc_original_wout_DP = eval_accuracy(all_label_probs_wout_DP, test_labels)
            acc_calibrated_wout_DP = eval_accuracy(
                all_label_probs_wout_DP,
                test_labels,
                mode="diagonal_W",
                p_cf=p_cf_wout_DP,
            )

            # print(f"Accuracy (original) without DP: {acc_original_wout_DP}")
            # print(f"Accuracy (calibrated) without DP: {acc_calibrated_wout_DP}")

            if use_dp_prompts:
                p_cf_w_DP = get_p_content_free(
                    synthetic_examples,
                    openai_model,
                    content_free_inputs=content_free_inputs,
                )

                all_raw_answers_w_DP = await get_model_response(
                    synthetic_examples, test_subset, openai_model
                )
                all_label_probs_w_DP = get_label_probs(
                    all_raw_answers_w_DP, test_subset
                )

                acc_original_w_DP = eval_accuracy(all_label_probs_w_DP, test_labels)
                acc_calibrated_w_DP = eval_accuracy(
                    all_label_probs_w_DP, test_labels, mode="diagonal_W", p_cf=p_cf_w_DP
                )

                # print(f"Accuracy (original) with DP: {acc_original_w_DP}")
                # print(f"Accuracy (calibrated) with DP: {acc_calibrated_w_DP}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise


def _parse_args():
    """Use stdlib argparse because Typer 0.9 is incompatible with Click 8.5."""
    parser = argparse.ArgumentParser(description="Run local ICL experiments")
    parser.add_argument("--sigma", type=float, default=0.39)
    parser.add_argument("--openai-model", default="chatglm3-6b")
    parser.add_argument("--task", choices=("sst2", "wikitext"), default="sst2")
    parser.add_argument("--data-dir", default="../data/wikitext_2_raw_v1")
    parser.add_argument(
        "--wikitext-query-split",
        choices=("train", "validation", "test"),
        default="validation",
        help="WikiText split used for continuation queries; use train for member extraction attacks.",
    )
    parser.add_argument("--wikitext-min-words", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--num-private-train", type=int, default=DEFAULT_NUM_PRIVATE_TRAIN)
    parser.add_argument("--set-num-public-train", type=int, default=DEFAULT_NUM_PUBLIC_TRAIN)
    parser.add_argument("--num-valid", type=int, default=DEFAULT_NUM_VALID)
    parser.add_argument("--num-private-train-splits", type=int, default=DEFAULT_NUM_PRIVATE_TRAIN_SPLITS)
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST)
    parser.add_argument("--no-public-token", action="store_true")
    parser.add_argument("--subsample-per-token", action="store_true")
    parser.add_argument("--use-dp-prompts", dest="use_dp_prompts", action="store_true")
    parser.add_argument("--no-use-dp-prompts", dest="use_dp_prompts", action="store_false")
    parser.set_defaults(use_dp_prompts=True)
    parser.add_argument("--sample-same-label-prompts", action="store_true")
    parser.add_argument("--normalize-max", action="store_true")
    parser.add_argument("--max-token-per-text", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--synth-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--no-noise", action="store_true", help="Disable all private-prompt noise for direct evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    _main(**vars(_parse_args()))
