"""Portable paths and explicit single-GPU model loading."""
from pathlib import Path
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DPICL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DPICL_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
MODEL_ALIASES = {
    "chatglm3": Path(os.environ.get(
        "CHATGLM_MODEL_DIR", WORKSPACE_DIR / "models/chatglm3-6b"
    )).expanduser().resolve(),
    "fm9g": Path(os.environ.get(
        "FM9G_MODEL_DIR", WORKSPACE_DIR / "models/9G7B_MHA"
    )).expanduser().resolve(),
}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def model_path(value):
    if value in MODEL_ALIASES:
        candidate = MODEL_ALIASES[value]
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        return str(candidate)
    candidate = project_path(value)
    if candidate.is_dir():
        return str(candidate)
    if value.startswith((".", "/", "~")):
        raise FileNotFoundError(candidate)
    return value


def select_device(gpu):
    if gpu not in range(4):
        raise ValueError("--gpu must be 0, 1, 2 or 3")
    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise RuntimeError(f"Requested cuda:{gpu} is unavailable; no CPU fallback")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def load_target(value, device, dtype="float16", trust_remote_code=False):
    name = model_path(value)
    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=False, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if device is not None and torch.device(device).type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
        if free < weight_bytes + 2 * 2**30:
            raise RuntimeError(f'Insufficient memory on {device}: free={free / 2**30:.1f} GiB, '
                               f'weights={weight_bytes / 2**30:.1f} GiB plus 2 GiB minimum reserve; '
                               'choose another --gpu or explicitly reduce --dtype precision')
    model = model.to(device).eval()
    return tokenizer, model


def load_sst2(directory=None):
    from datasets import Dataset, DatasetDict, load_dataset
    if directory is None:
        return load_dataset('glue', 'sst2')
    import pandas as pd
    path = project_path(directory)
    return DatasetDict({split: Dataset.from_pandas(pd.read_csv(path / filename, sep='\t'),
                                                  preserve_index=False)
                        for split, filename in [('train', 'train.tsv'),
                                                ('validation', 'dev.tsv'), ('test', 'test.tsv')]})
