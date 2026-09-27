"""Original neighbourhood attack with portable configuration and saved provenance."""
import argparse
from heapq import nlargest
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from runtime_config import load_target, model_path, project_path, select_device
from task_data import load_wikitext2, text_key, WIKITEXT_CONFIGS

logger = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proc-id', type=int, default=0)
    parser.add_argument('--model', choices=['bert', 'distilbert', 'roberta'], default='bert')
    parser.add_argument('--proposal-model', help='Local directory or HF ID; overrides --model default')
    parser.add_argument('--dataset', choices=['sst2', 'wikitext2', 'other'], required=True)
    parser.add_argument('--task-dataset', choices=['sst2', 'wikitext2'], default='sst2')
    parser.add_argument('--wikitext2-dir')
    parser.add_argument('--wikitext2-config', choices=WIKITEXT_CONFIGS, default=WIKITEXT_CONFIGS[0])
    parser.add_argument('--target-model', default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float32')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--input-json', default='DPICL/Indirect_attack/synthetic_text_results.json')
    parser.add_argument('--prompt-field', choices=['original_text', 'synthetic_text'], default='synthetic_text')
    parser.add_argument('--prompt-start', type=int, default=5)
    parser.add_argument('--prompt-stop', type=int, default=10)
    parser.add_argument('--output-dir', default='runs/attack')
    parser.add_argument('--batch-size', type=int, default=500, help='Samples per process slice, not model batch size')
    parser.add_argument('--num-neighbors', type=int, default=10)
    parser.add_argument('--neighbor-top-k', type=int, default=6)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--proposal-max-length', type=int,
                        help='Proposal tokenizer limit; defaults to target limit for SST-2, 512 for WikiText-2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, help='Cap samples after selecting the process slice')
    parser.add_argument('--nonmember-file', help='Optional JSON list of strings/records or CSV with sentence/text column')
    parser.add_argument('--score-mode', choices=['legacy', 'token-mean'], default='token-mean',
                        help='legacy preserves original padding treatment; token-mean masks padding and uses attention masks')
    return parser


def build_prompt(data, field, start=5, stop=10, task_dataset='sst2'):
    chosen = data[start:stop]
    if not chosen:
        raise ValueError('Prompt selection is empty; check --prompt-start/--prompt-stop')
    if task_dataset == 'wikitext2':
        lines = [f"Text:{item[field]}\n" for item in chosen]
    else:
        lines = [f"Text:{item[field]}.Label:{item['label']}\n" for item in chosen]
    return 'Example:\n' + ''.join(lines) + '\nPlease analyze this text:'


def load_nonmembers(value=None, dataset='sst2', directory=None, config=WIKITEXT_CONFIGS[0]):
    if dataset == 'wikitext2' and value is None:
        data, _ = load_wikitext2(directory, config)
        return list(data['test']['sentence'])
    if value is None:
        texts = load_dataset('glue', 'sst2')['test']['sentence']
    else:
        path = project_path(value)
        if path.suffix == '.json':
            entries = json.loads(path.read_text(encoding='utf-8'))
            texts = [x if isinstance(x, str) else x.get('sentence', x.get('text')) for x in entries]
        else:
            table = pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',')
            texts = table['sentence' if 'sentence' in table else 'text'].tolist()
    if any(not isinstance(t, str) for t in texts):
        raise ValueError('Nonmember texts must be strings')
    return [t for t in texts if len(t) > 10]


def generate_neighbours(text, tokenizer, model, device, top_k=6, max_length=512):
    ids = tokenizer(text, truncation=True, max_length=max_length,
                    return_tensors='pt').input_ids.to(device)
    replacements = {}
    dropout = torch.nn.Dropout(p=0.7)
    # Preserve the author's positions, embedding dropout and top-50 single replacements.
    backbone = getattr(model, model.base_model_prefix)
    with torch.no_grad():
        for index in range(1, ids.shape[1]):
            original = ids[0, index]
            embeds = backbone.embeddings(ids)
            embeds = torch.cat((embeds[:, :index], dropout(embeds[:, index]).unsqueeze(1),
                                embeds[:, index + 1:]), dim=1)
            probs = torch.softmax(model(inputs_embeds=embeds).logits, dim=2)
            original_prob = probs[0, index, original].item()
            values, candidates = torch.topk(probs[:, index, :], top_k, dim=1)
            for candidate, value in zip(candidates[0], values[0]):
                if candidate != original:
                    denominator = 0.1 if original_prob == 1 else 1 - original_prob
                    replacements[(index, int(candidate))] = value.item() / denominator
    keys = nlargest(50, replacements, key=replacements.get)
    result = []
    for index, candidate in keys:
        alt = ids.clone()
        alt[0, index] = candidate
        result.append(tokenizer.batch_decode(alt)[0])
    return result


def clean_bert_text(text):
    return text.replace(' [SEP]', ' ').replace('[CLS] ', ' ')


def score_texts(prompt, texts, tokenizer, model, device, max_length=512,
                mode='token-mean', original=False):
    encoded = tokenizer([prompt + t for t in texts], padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
    prompt_len = tokenizer(prompt, truncation=True, max_length=max_length,
                           return_tensors='pt').input_ids.shape[1]
    ids = encoded.input_ids
    if mode == 'legacy' and original:
        labels = ids.clone()
        labels[:, :prompt_len] = -100
        if not (labels[:, 1:] != -100).any():
            raise ValueError('Prompt consumed the entire max-length; no target tokens remain')
        with torch.no_grad():
            return [-model(ids, labels=labels).loss.item()]
    mask = torch.ones_like(ids[:, 1:], dtype=torch.bool)
    mask[:, :prompt_len - 1] = False
    kwargs = {}
    if mode == 'token-mean':
        kwargs['attention_mask'] = encoded.attention_mask
        positions = encoded.attention_mask.cumsum(1) - 1
        kwargs['position_ids'] = positions.clamp_min(0)
        prefix = tokenizer(prompt, truncation=True, max_length=max_length,
                           return_tensors='pt').input_ids[0].tolist()
        starts = []
        # A token spanning the prompt/body boundary belongs to the target text.
        for row, attention in zip(ids, encoded.attention_mask):
            valid = row[attention.bool()].tolist()
            start = 0
            for before, after in zip(prefix, valid):
                if before != after:
                    break
                start += 1
            starts.append(start)
        starts = torch.tensor(starts, device=ids.device).unsqueeze(1)
        mask = (encoded.attention_mask[:, 1:].bool()
                & encoded.attention_mask[:, :-1].bool()
                & (positions[:, 1:] >= starts))
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError('No target tokens remain; shorten the prompt or increase --max-length')
    with torch.no_grad():
        logits = model(ids, use_cache=False, **kwargs).logits[:, :-1, :]
        if mode == 'token-mean':
            logits = logits.float()
        if mode == 'token-mean':
            # Match native LM loss; spatial CE accumulates drift for large vocabularies.
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none'
            ).reshape(ids.shape[0], -1)
        else:
            losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), ids[:, 1:], reduction='none')
        losses = losses.masked_fill(~mask, 0)
        scores = -losses.sum(1) / counts
    return scores.cpu().tolist()


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if min(args.batch_size, args.num_neighbors, args.neighbor_top_k, args.max_length) < 1 or args.proc_id < 0:
        raise ValueError('Sizes must be positive and --proc-id must be nonnegative')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.dataset == 'wikitext2':
        args.task_dataset = 'wikitext2'
    if args.dataset == 'sst2' and args.task_dataset != 'sst2':
        raise ValueError('Nonmember --dataset must match --task-dataset')
    if args.proposal_max_length is not None and args.proposal_max_length < 1:
        raise ValueError('--proposal-max-length must be positive')
    proposal_length = args.proposal_max_length or (512 if args.task_dataset == 'wikitext2' else args.max_length)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.gpu)
    data = json.loads(project_path(args.input_json).read_text(encoding='utf-8'))
    if any(x.get('task_dataset', 'sst2') != args.task_dataset for x in data):
        raise ValueError('Demonstration JSON does not match --task-dataset')
    prompt = build_prompt(data, args.prompt_field, args.prompt_start, args.prompt_stop, args.task_dataset)
    texts = ([x['original_text'] for x in data] if args.dataset == 'other' else
             load_nonmembers(args.nonmember_file, args.dataset, args.wikitext2_dir, args.wikitext2_config))
    offset = args.proc_id * args.batch_size
    texts = texts[offset:offset + args.batch_size]
    if args.limit is not None:
        texts = texts[:args.limit]
    if not texts:
        raise ValueError('Requested sample slice is empty')
    if args.task_dataset == 'wikitext2' and args.limit is not None and len(texts) < args.limit:
        raise ValueError(f'Requested {args.limit} samples, but only {len(texts)} are available')
    if args.task_dataset == 'wikitext2':
        wiki_data, _ = load_wikitext2(args.wikitext2_dir, args.wikitext2_config)
        private_keys = {text_key(t) for t in wiki_data['train']['sentence']}
        keys = [text_key(t) for t in texts]
        if len(keys) != len(set(keys)):
            raise ValueError('WikiText-2 attack samples contain duplicates')
        if args.dataset == 'other' and not set(keys) <= private_keys:
            raise ValueError('WikiText-2 member texts must belong to the private generation pool')
        if args.dataset != 'other' and set(keys) & private_keys:
            raise ValueError('WikiText-2 nonmembers overlap the private generation pool')
    destination = project_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f'attack_results_{args.dataset}_{args.model}_{args.proc_id}_{args.prompt_field}'
    output = destination / f'{stem}.csv'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    config = vars(args).copy()
    config.update(target_model=model_path(args.target_model), input_json=str(project_path(args.input_json)),
                  requested_samples=len(texts), prompt=prompt, model_adapter_version=2)
    config['effective_proposal_max_length'] = proposal_length
    (destination / f'{stem}.config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if args.score_mode == 'legacy':
        logger.warning('Legacy scoring preserves author padding handling; use --score-mode token-mean for corrected scoring')
    logger.info('Target and proposal both use %s; dataset=%s, samples=%s', device, args.dataset, len(texts))
    tokenizer, target = load_target(args.target_model, device, args.dtype, args.trust_remote_code)
    if args.task_dataset == 'wikitext2':
        prefix_length = len(tokenizer(prompt).input_ids)
        if prefix_length >= args.max_length - 1:
            raise ValueError(f'WikiText-2 prompt has {prefix_length} tokens; increase --max-length '
                             f'above {prefix_length + 1}; prompt examples are not silently dropped')
    proposal_name = args.proposal_model or {'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased', 'roberta': 'roberta-base'}[args.model]
    proposal_name = model_path(proposal_name)
    search_tokenizer = AutoTokenizer.from_pretrained(proposal_name, use_fast=False)
    search_model = AutoModelForMaskedLM.from_pretrained(proposal_name).to(device).eval()
    if args.task_dataset == 'wikitext2':
        proposal_limit = getattr(search_model.config, 'max_position_embeddings', None)
        if proposal_limit is not None and proposal_length > proposal_limit:
            raise ValueError(f'--proposal-max-length exceeds the proposal model limit {proposal_limit}')
    rows, failed = [], []
    for index, text in enumerate(tqdm(texts, desc=f'{args.dataset}/{args.prompt_field}')):
        try:
            ids = search_tokenizer(text, truncation=True, max_length=proposal_length,
                                   return_tensors='pt').input_ids
            original = clean_bert_text(search_tokenizer.batch_decode(ids)[0])
            orig_score = score_texts(prompt, [original], tokenizer, target, device,
                                    args.max_length, args.score_mode, original=True)[0]
            neighbours = generate_neighbours(text, search_tokenizer, search_model, device,
                                             args.neighbor_top_k, proposal_length)
            if not neighbours:
                raise ValueError('No neighbours generated')
            neighbours = [clean_bert_text(t) for t in neighbours[:args.num_neighbors]]
            scores = score_texts(prompt, neighbours, tokenizer, target, device,
                                 args.max_length, args.score_mode)
            mean = sum(scores) / len(scores)
            if not np.isfinite([orig_score, mean, *scores]).all():
                raise ValueError('Nonfinite target score')
            rows.append(dict(index=index, text=original, orig_logprob=orig_score,
                             mean_neigh_logprob=mean, diff_score=orig_score - mean))
        except Exception as exc:
            logger.warning('Text %s failed: %s', offset + index, exc)
            failed.append(dict(index=offset + index, text=text, error=str(exc)))
    pd.DataFrame(rows, columns=['index', 'text', 'orig_logprob', 'mean_neigh_logprob', 'diff_score']).sort_values(
        'diff_score', ascending=False).to_csv(output, index=False, encoding='utf-8-sig')
    (destination / f'{stem}.failed.json').write_text(json.dumps(failed, indent=2), encoding='utf-8')
    logger.info('Saved %s: %s successful, %s failed (not included in CSV)', output, len(rows), len(failed))
    if failed:
        raise RuntimeError('Attack finished with failed samples; inspect failed.json before evaluating')
    return output


if __name__ == '__main__':
    main()
