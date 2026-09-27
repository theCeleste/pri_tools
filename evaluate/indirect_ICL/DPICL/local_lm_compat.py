# local_lm_compat.py
# Compatibility shim that exposes the minimal functions expected by run_exp_agnews.py
# - api_openai_com(model_name) -> LocalLM instance
# - next_logprobs(lm, prompt_token_ids, top_p=1.0) -> dict token_id -> logprob
# - normalized_logprobs_for_chosen_tokens(lm, prompt_token_ids, chosen_tokens_iterable, top_p=1.0) -> dict

from local_llm import LocalLM
from typing import Iterable, Dict
import torch
import asyncio
from runtime_config import model_path

# ==== GLOBAL MODEL CACHE ====
_LM_CACHE = {}

def api_openai_com(model_name: str, device: torch.device = None,
                   dtype=None, trust_remote_code=None) -> LocalLM:
    """
    Return a cached LocalLM instance.
    Ensures the LLaMA model is only loaded ONCE.
    """
    global _LM_CACHE
    model_name = model_path(model_name)
    if device is None:
        matches = [value for key, value in _LM_CACHE.items() if key[0] == model_name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError('Multiple cached devices for this model; pass device explicitly')
    dtype = dtype or torch.float16
    trust_remote_code = bool(trust_remote_code)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    key = (model_name, str(device), str(dtype), trust_remote_code)
    if key in _LM_CACHE:
        return _LM_CACHE[key]

    # otherwise load once
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lm = LocalLM(model_name, device=device, dtype=dtype, trust_remote_code=trust_remote_code)

    _LM_CACHE[key] = lm
    return lm

async def next_logprobs(lm: LocalLM, prompt_token_ids: list, top_p: float = 1.0):
    """
    Async wrapper to compute next-token logprobs for a prompt (token ids).
    Returns a python dict mapping token_id -> logprob (float).
    Mirrors the expected behavior of the original next_logprobs.
    """
    # Here we simply call lm.get_next_token_logprobs (sync) and return result in async context
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: lm.get_next_token_logprobs(prompt_token_ids, top_k=100, top_p=top_p))

async def normalized_logprobs_for_chosen_tokens(lm: LocalLM, prompt_token_ids: list, chosen_tokens_iterable: Iterable[int], top_p: float = 1.0):
    """
    Async wrapper returning logprobs for the chosen token ids only.
    Mirrors normalized_logprobs_for_chosen_tokens semantics from the original code.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: lm.normalized_logprobs_for_chosen_tokens(prompt_token_ids, chosen_tokens_iterable, top_p=top_p))
