# local_llm.py
# Local HuggingFace causal-LM wrapper used by the ICL experiment.
# - fp16 (where available)
# - token alignment utilities (encode single token, encode/decode)
# - next-token logprobs (top-k / top-p support)
# - normalized logprobs for a chosen vocab subset
# - async completions(...) with similar shape to original code (list of token objects)

import os
import json
# Transformers copies local `trust_remote_code` modules into this cache.  The
# cluster home cache can be read-only, so keep the dynamic modules beside the
# experiment code instead.
_MODULE_CACHE = os.path.join(os.path.dirname(__file__), ".hf_modules")
os.makedirs(_MODULE_CACHE, exist_ok=True)
os.environ.setdefault("HF_MODULES_CACHE", _MODULE_CACHE)
import torch
import math
import asyncio
from typing import List, Dict, Iterable, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch.nn.functional as F

class Encoding:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.n_vocab = len(tokenizer)
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_prompt(self, text: str) -> List[int]:
        """Encode one user prompt using the model's native conversation format."""
        return self.owner.encode_prompt(text)
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    def encode_single_token(self, text: str) -> Optional[int]:
        # Ensure the provided text maps to exactly one token
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        if len(toks) == 1:
            return toks[0]
        # If not single token, return first token (caller must be aware)
        return toks[0] if len(toks) > 0 else None

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
    def __init__(self, model_name_or_path: str, device: torch.device = None, dtype=None):
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.model_family = self._detect_model_family(model_name_or_path)
        if dtype is None:
            if device is not None and device.type == "cuda":
                use_bfloat16 = self.model_family == "9g" and torch.cuda.is_bf16_supported()
                dtype = torch.bfloat16 if use_bfloat16 else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        # ensure pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        configured_eos = getattr(self.tokenizer, "eos_token_id", None)
        if self.model_family == "chatglm":
            # ``<|user|>`` is a turn marker, not an assistant completion EOS.
            # Treating it as EOS causes ChatGLM to stop immediately on many
            # text-continuation prompts.  ChatGLM's configured ``</s>`` is the
            # only stop token needed here.
            self.eos_token_ids = {configured_eos} if configured_eos is not None else set()
        elif self.model_family == "9g":
            self.eos_token_ids = {x for x in (configured_eos, 73440) if x is not None}
        else:
            self.eos_token_ids = {configured_eos} if configured_eos is not None else set()

        # set model
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        # FM9G's bundled remote class uses the pre-Transformers-4.57
        # constructor contract and rejects the newer ``dtype`` keyword.
        # ``torch_dtype`` remains accepted by both the legacy remote class
        # and current Transformers (with only a deprecation warning).
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=self.dtype,
                **model_kwargs,
            ).to(self.device)
        except TypeError as exc:
            if "unexpected keyword argument 'torch_dtype'" not in str(exc):
                raise
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=self.dtype,
                **model_kwargs,
            ).to(self.device)

        # move to device if not automatically placed
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()

        self.encoding = Encoding(self.tokenizer)
        self.encoding.owner = self
        self.n_vocab = self.encoding.n_vocab
        self.tokenizer_model_max_length = self.tokenizer.model_max_length

    @staticmethod
    def _detect_model_family(model_name_or_path: str) -> str:
        """Identify local model families without relying on a remote model id."""
        config_path = os.path.join(model_name_or_path, "config.json")
        model_type = ""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type", "")
        except (OSError, ValueError, TypeError):
            pass
        name = os.path.basename(os.path.normpath(model_name_or_path)).lower()
        if model_type == "chatglm" or "chatglm" in name:
            return "chatglm"
        if model_type == "fm9g7b" or name.startswith("9g") or "9g" in name:
            return "9g"
        return "llama"

    def encode_prompt(self, text: str) -> List[int]:
        """Return prompt ids with a generation-ready turn for chat models."""
        if self.model_family == "llama":
            return self.encoding.encode(text)
        messages = [{"role": "user", "content": text}]
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            return list(ids)
        except (AttributeError, TypeError, ValueError):
            # Older ChatGLM tokenizers expose build_chat_input instead of the
            # transformers chat-template API.
            if self.model_family == "chatglm" and hasattr(self.tokenizer, "build_chat_input"):
                batch = self.tokenizer.build_chat_input(text, history=[])
                return batch["input_ids"][0].tolist()
            return self.encoding.encode(text)

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

        # This wrapper recomputes logits from the complete prompt on every
        # step, so KV caching provides no benefit here.  The bundled 9G
        # remote-code model expects the pre-4.43 Cache API; disabling cache
        # keeps it compatible with current Transformers (including 4.57.x)
        # without changing the generated-token behavior.
        outputs = self.model(input_ids=ids, return_dict=True, use_cache=False)
        logits = outputs.logits  # shape [1, seq_len, vocab]
        next_logits = logits[:, -1, :].squeeze(0)  # [vocab]
        return next_logits

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
                first_cutoff = cutoff[0, 1].item()
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
            prompt_ids = self.encode_prompt(input_ids_or_text)
        elif isinstance(input_ids_or_text, list):
            prompt_ids = input_ids_or_text
        else:
            raise ValueError("Unsupported prompt type for completions")

        max_tokens = getattr(settings, "max_tokens", 50)
        min_tokens = max(0, int(getattr(settings, "min_tokens", 0)))
        stop_tokens = getattr(settings, "stop", None)
        temperature = getattr(settings, "temperature", 0.0)
        n = getattr(settings, "n", 1)

        # simple greedy / temperature-sampled sampling
        generated = []
        cur_ids = prompt_ids.copy()
        for _ in range(max_tokens):
            logits = self._get_logits(cur_ids)
            if len(generated) < min_tokens and self.eos_token_ids:
                logits = logits.clone()
                for eos_token_id in self.eos_token_ids:
                    if 0 <= eos_token_id < logits.shape[-1]:
                        logits[eos_token_id] = -float("inf")
            if temperature and temperature > 0.0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(torch.argmax(logits).item())
            cur_ids.append(next_token)
            generated.append(next_token)
            if next_token in self.eos_token_ids:
                break
            # check stop tokens if provided as strings: encode and check suffix
            if stop_tokens:
                decoded = self.encoding.decode(generated[-len(max(stop_tokens, key=len)) :])
                # we won't do robust stop detection here; leave for caller
        # Return a single completion object compatible with original usage:
        # For simplicity, return each token wrapped as SimpleToken-like
        return [[SimpleToken(tid) for tid in generated]]

    def decode(self, token_ids: List[int]) -> str:
        return self.encoding.decode(token_ids)
