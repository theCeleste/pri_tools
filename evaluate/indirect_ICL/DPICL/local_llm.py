# local_llm.py
# Local HuggingFace Llama wrapper that mimics the minimal behavior expected by run_exp_agnews.py
# - fp16 (where available)
# - token alignment utilities (encode single token, encode/decode)
# - next-token logprobs (top-k / top-p support)
# - normalized logprobs for a chosen vocab subset
# - async completions(...) with similar shape to original code (list of token objects)

import torch
import math
import asyncio
from typing import List, Dict, Iterable, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch.nn.functional as F
from runtime_config import load_target

class Encoding:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.n_vocab = len(tokenizer)
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    def encode_single_token(self, text: str) -> Optional[int]:
        # Ensure the provided text maps to exactly one token
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        if len(toks) == 1:
            return toks[0]
        return None

    def trim_stops(self, token_ids, stop_strings):
        text = self.decode(token_ids)
        ends = [text.find(stop) for stop in stop_strings if stop and stop in text]
        return text[:min(ends)] if ends else text

    def reached_stop(self, token_ids, stop_strings):
        text = self.decode(token_ids)
        return any(stop and stop in text for stop in stop_strings)

    def stop_token_ids(self, stop_strings):
        return {token for text in stop_strings
                if (token := self.encode_single_token(text)) is not None}

class SimpleToken:
    def __init__(self, token_id: int):
        self.token = type("T", (), {"token_id": token_id})

class LocalLM:
    """
    LocalLM: a wrapper around HF causal LM to provide a small set of functions:
      - encoding.encode / decode / encode_single_token
      - n_vocab
      - get_next_token_logprobs(input_ids, top_k=None, top_p=1.0) -> Dict[int, float] (logprobs)
      - normalized_logprobs_for_chosen_tokens(input_ids, chosen_tokens_iterable, top_p=1.0)
      - get_logprob_of_target(prompt_text, target_text)
      - async completions(input_ids_or_text, settings) -> list of SimpleToken objects (mimic)
    """
    @property
    def eos_token_ids(self):
        result = set()
        model = getattr(self, 'model', None)
        for source in [self.tokenizer, getattr(model, 'config', None),
                       getattr(model, 'generation_config', None)]:
            value = getattr(source, 'eos_token_id', None)
            if value is not None:
                result.update(value if isinstance(value, (list, tuple)) else [value])
        return result

    @property
    def input_prefix_tokens(self):
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            return self.tokenizer.get_prefix_tokens()
        if getattr(self.tokenizer, 'add_bos_token', False):
            return [self.tokenizer.bos_token_id]
        return []

    def __init__(self, model_name_or_path: str, device: torch.device = None, dtype=torch.float16,
                 trust_remote_code=False):
        self.device = device
        self.dtype = dtype

        # load tokenizer and model
        self.tokenizer, self.model = load_target(
            model_name_or_path, device, str(dtype).split(".")[-1], trust_remote_code
        )
        # ensure pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # set model
        # move to device if not automatically placed
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.encoding = Encoding(self.tokenizer)
        self.n_vocab = self.encoding.n_vocab
        self.tokenizer_model_max_length = self.tokenizer.model_max_length

    @torch.no_grad()
    def _get_logits(self, input_ids: List[int]) -> torch.Tensor:
        """
        Return logits for next token (shape: [vocab_size])
        """
        if isinstance(input_ids, list):
            ids = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        elif isinstance(input_ids, torch.Tensor):
            ids = input_ids.to(self.device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
        else:
            raise ValueError("input_ids must be list[int] or torch.Tensor")

        prefix = self.input_prefix_tokens
        if prefix:
            prefix_ids = torch.tensor(prefix, device=ids.device, dtype=ids.dtype)
            present = (ids[:, :len(prefix)] == prefix_ids).all(1) if ids.shape[1] >= len(prefix) else torch.zeros(
                ids.shape[0], device=ids.device, dtype=torch.bool)
            if present.any() and not present.all():
                raise ValueError('Mixed prefixed/unprefixed inputs; tokenize each input consistently')
            if not present.all():
                ids = torch.cat((prefix_ids.expand(ids.shape[0], -1), ids), dim=1)

        outputs = self.model(input_ids=ids, return_dict=True)
        logits = outputs.logits  # shape [1, seq_len, vocab]
        next_logits = logits[:, -1, :].squeeze(0)  # [vocab]
        return next_logits[:self.encoding.n_vocab]

    def _logsoftmax(self, logits: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            lsm = F.log_softmax(logits, dim=-1)
            return lsm.cpu().numpy()

    def _apply_top_k_top_p(self, logits: torch.Tensor, top_k: Optional[int], top_p: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (token_ids, logprobs) after applying top_k/top_p filter (top-p nucleus).
        """
        logits = logits.clone()
        # top-k
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k)
            mask = torch.full_like(logits, -float("inf"))
            mask[topk_idx] = logits[topk_idx]
            logits = mask

        # top-p
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # keep tokens with cumulative_probs <= top_p
            cutoff = (cumulative_probs > top_p).nonzero()
            if cutoff.nelement() != 0:
                first_cutoff = cutoff[0, 0].item()
                # mask all tokens after first_cutoff in sorted order
                indices_to_remove = sorted_indices[first_cutoff + 1 :]
                logits[indices_to_remove] = -float("inf")

        logprobs = F.log_softmax(logits, dim=-1)
        # select entries that are finite
        finite_mask = torch.isfinite(logprobs)
        token_ids = finite_mask.nonzero(as_tuple=False)[:, 0].cpu().numpy()
        token_logprobs = logprobs[finite_mask].cpu().numpy()
        return token_ids, token_logprobs

    def get_next_token_logprobs(self, input_ids: List[int], top_k: Optional[int] = 100, top_p: float = 1.0) -> Dict[int, float]:
        """
        Return a dict mapping token_id -> logprob for next token.
        By default returns top_k tokens (100) to mimic original top-k trimming.
        """
        logits = self._get_logits(input_ids)  # torch.Tensor [vocab]
        # apply top_k/top_p trimming
        if top_k is None and (top_p is None or top_p >= 1.0):
            # return full distribution
            lsm = self._logsoftmax(logits)
            return {int(i): float(lsm[i]) for i in range(lsm.shape[0])}
        else:
            token_ids, token_logprobs = self._apply_top_k_top_p(logits, top_k, top_p)
            return {int(t): float(lp) for t, lp in zip(token_ids.tolist(), token_logprobs.tolist())}

    def normalized_logprobs_for_chosen_tokens(self, input_ids: List[int], chosen_tokens: Iterable[int], top_p: float = 1.0) -> Dict[int, float]:
        """
        Return logprobs for the chosen_tokens only (not normalized across whole vocab).
        We follow original: compute log-softmax over full logits, then extract chosen tokens' logprobs.
        Optionally apply top-p truncation to the logits before normalization to mimic top-k filtering.
        """
        logits = self._get_logits(input_ids)
        # optionally restrict logits to chosen tokens by setting others to -inf (but better to compute full lsm)
        lsm = F.log_softmax(logits, dim=-1)
        lsm_np = lsm.cpu().numpy()
        out = {}
        for t in chosen_tokens:
            if 0 <= int(t) < lsm_np.shape[0]:
                out[int(t)] = float(lsm_np[int(t)])
        return out

    def get_logprob_of_target(self, prompt_text: str, target_text: str) -> float:
        """
        Compute log-probability of target_text given prompt_text.
        If target_text tokenizes to multiple tokens, we compute conditional logprob sum:
          log p(t1 | prompt) + log p(t2 | prompt + t1) + ...
        Returns a scalar float (log prob).
        """
        prompt_ids = self.encoding.encode(prompt_text)
        target_ids = self.encoding.encode(target_text)
        total_logprob = 0.0
        cur_ids = prompt_ids.copy()
        for tid in target_ids:
            lps = self.get_next_token_logprobs(cur_ids, top_k=None, top_p=1.0)
            # lps contains full vocab logprobs (because top_k=None)
            if tid in lps:
                total_logprob += float(lps[tid])
            else:
                # if tid missing (shouldn't), approximate with very small prob
                total_logprob += -100.0
            cur_ids.append(tid)
        return float(total_logprob)

    async def completions(self, input_ids_or_text, settings):
        """
        Minimal async completions compatibility.
        settings expected to have .temperature, .max_tokens, .n, .stop (list)
        Return: a list with one element: completion tokens in a simple structure
        We'll implement greedy generation (temperature ignored unless !=0)
        """
        # Accept either token ids list or raw text
        if isinstance(input_ids_or_text, str):
            prompt_ids = self.encoding.encode(input_ids_or_text)
        elif isinstance(input_ids_or_text, list):
            prompt_ids = input_ids_or_text
        else:
            raise ValueError("Unsupported prompt type for completions")

        max_tokens = getattr(settings, "max_tokens", 50)
        stop_tokens = getattr(settings, "stop", None)
        stop_ids = self.eos_token_ids | self.encoding.stop_token_ids(stop_tokens or [])
        temperature = getattr(settings, "temperature", 0.0)
        n = getattr(settings, "n", 1)

        # simple greedy / temperature-sampled sampling
        generated = []
        cur_ids = prompt_ids.copy()
        for _ in range(max_tokens):
            logits = self._get_logits(cur_ids)
            if temperature and temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(logits).item())
            if next_token in stop_ids:
                break
            cur_ids.append(next_token)
            generated.append(next_token)
            if stop_tokens and self.encoding.reached_stop(generated, stop_tokens):
                text = self.encoding.trim_stops(generated, stop_tokens)
                generated = self.encoding.encode(text)
                break
        # Return a single completion object compatible with original usage:
        # For simplicity, return each token wrapped as SimpleToken-like
        return [[SimpleToken(tid) for tid in generated]]

    def decode(self, token_ids: List[int]) -> str:
        return self.encoding.decode(token_ids)
