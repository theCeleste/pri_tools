"""Fixed-prompt, label-likelihood Subj holdout accuracy for every Glm-subj variant."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Glm-subj'))
from privacy_attack.common import load_split, validate_adapter, validate_chatglm_base, write_json

PROMPT = ('Is the following sentence subjective or objective?\n\nSentence: "{text}"\n\n'
          'Options:\n(A) The sentence is subjective.\n(B) The sentence is objective.')
LABELS = ('(B) The sentence is objective.', '(A) The sentence is subjective.')


def holdout(data_dir, max_samples):
    rows = load_split(data_dir, 'test')
    if max_samples < 0:
        raise ValueError('max-samples must be nonnegative')
    rows = rows[:max_samples] if max_samples else rows
    if not rows or any(row.get('label') not in (0, 1) for row in rows):
        raise ValueError('Subj holdout must contain binary labels')
    digest = hashlib.sha256(json.dumps([(r['text'], int(r['label'])) for r in rows],
                                       ensure_ascii=False).encode()).hexdigest()
    return rows, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-model', 'adapter', 'data-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steering')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    args = parser.parse_args(argv)
    validate_chatglm_base(args.base_model)
    validate_adapter(args.adapter)
    rows, digest = holdout(args.data_dir, args.max_samples)
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Subj utility evaluation requires CUDA')
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.encode_special_tokens = True
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, trust_remote_code=True,
                                               local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter).to('cuda:0').eval()
    context = nullcontext()
    if args.steering:
        from privacy_attack.steering import load_steering, apply_steering
        metadata, directions = load_steering(args.steering)
        context = apply_steering(model, directions, float(metadata['inference_multiplier']))
    records = []
    with context, torch.inference_mode():
        for row in rows:
            prefix = '<|user|>\n' + PROMPT.format(text=row['text']) + '<|assistant|>\n'
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            scores = []
            for label in LABELS:
                # Tokenize continuation separately so the scored span is identical across labels.
                suffix_ids = tokenizer.encode(label, add_special_tokens=False)
                ids = torch.tensor([prefix_ids + suffix_ids], device='cuda:0')
                logits = model(input_ids=ids).logits
                if logits.shape[0] != 1:
                    logits = logits.transpose(0, 1)
                start = len(prefix_ids) - 1
                log_probs = logits[0, start:-1].float().log_softmax(-1)
                targets = ids[0, len(prefix_ids):]
                scores.append(float(log_probs.gather(-1, targets[:, None]).mean()))
            prediction = int(scores[1] > scores[0])
            records.append(dict(label=int(row['label']), prediction=prediction, scores=scores))
    correct = sum(r['label'] == r['prediction'] for r in records)
    write_json(args.output, dict(metric='accuracy', evaluator='subj_label_mean_log_likelihood_v1',
                                prompt=PROMPT, labels=LABELS, split='test', holdout_sha256=digest,
                                correct=correct, total=len(records), accuracy=correct / len(records),
                                configuration=vars(args), records=records))


if __name__ == '__main__':
    main()
