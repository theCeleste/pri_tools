import logging
from types import MethodType

import torch
from torch import nn


logger = logging.getLogger(__name__)


class PrefixLayer(nn.Module):
    def __init__(self, num_key_value_heads, num_prefix, head_dim, device, dtype):
        super().__init__()
        shape = (num_key_value_heads, num_prefix, head_dim)
        self.key = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        self.value = nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        nn.init.normal_(self.key, mean=0.0, std=0.02)
        nn.init.normal_(self.value, mean=0.0, std=0.02)

    def forward(self, batch_size):
        return (
            self.key.unsqueeze(0).expand(batch_size, -1, -1, -1),
            self.value.unsqueeze(0).expand(batch_size, -1, -1, -1),
        )


class LlamaPrefixEncoder(nn.Module):
    """Direct per-layer KV prefixes for the legacy Llama cache API."""

    def __init__(self, model, num_prefix):
        super().__init__()
        config = model.config
        self.num_prefix = num_prefix
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        layers = model.model.layers
        self.layers = nn.ModuleList()
        for layer in layers:
            projection = layer.self_attn.k_proj
            self.layers.append(
                PrefixLayer(
                    num_kv_heads,
                    num_prefix,
                    head_dim,
                    projection.weight.device,
                    projection.weight.dtype,
                )
            )

    def forward(self, batch_size):
        return tuple(layer(batch_size) for layer in self.layers)

    def initialize_from_cache(self, past_key_values):
        if len(past_key_values) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache layers, received {len(past_key_values)}"
            )
        with torch.no_grad():
            for layer, (key, value) in zip(self.layers, past_key_values):
                cached_key = key[0].to(device=layer.key.device, dtype=layer.key.dtype)
                cached_value = value[0].to(device=layer.value.device, dtype=layer.value.dtype)
                if cached_key.shape != layer.key.shape or cached_value.shape != layer.value.shape:
                    raise ValueError(
                        "Llama cache shape does not match prefix parameters: "
                        f"key={tuple(cached_key.shape)} expected={tuple(layer.key.shape)}, "
                        f"value={tuple(cached_value.shape)} expected={tuple(layer.value.shape)}"
                    )
                layer.key.copy_(cached_key)
                layer.value.copy_(cached_value)


def _prefix_forward(self, input_ids=None, attention_mask=None, past_key_values=None, inputs_embeds=None, **kwargs):
    if past_key_values is None:
        source = input_ids if input_ids is not None else inputs_embeds
        if source is None:
            raise ValueError("Prefix tuning requires input_ids or inputs_embeds")
        batch_size, sequence_length = source.shape[:2]
        past_key_values = self.prefix_encoder(batch_size)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), dtype=torch.long, device=source.device
            )
        if kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids + self.prefix_encoder.num_prefix
        prefix_mask = torch.ones(
            (batch_size, self.prefix_encoder.num_prefix),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
    return self.prefix_original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def _prefix_prepare_inputs_for_generation(self, *args, **kwargs):
    model_inputs = self.prefix_original_prepare_inputs_for_generation(*args, **kwargs)
    past_key_values = model_inputs.get("past_key_values")
    attention_mask = model_inputs.get("attention_mask")
    if past_key_values is not None and attention_mask is not None:
        current_length = 1
        if model_inputs.get("input_ids") is not None:
            current_length = model_inputs["input_ids"].shape[-1]
        past_length = past_key_values[0][0].shape[-2]
        expected_length = past_length + current_length
        missing = expected_length - attention_mask.shape[-1]
        if missing > 0:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], missing),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            model_inputs["attention_mask"] = torch.cat((prefix_mask, attention_mask), dim=-1)
            # The original generation helper derived positions before it knew about
            # the prefix. Keep generated tokens on the same position sequence used
            # by the initial prefixed forward pass.
            if model_inputs.get("position_ids") is not None:
                model_inputs["position_ids"] = model_inputs["position_ids"] + missing
    return model_inputs


class PrefixTuning:
    def __init__(self, model, num_prefix, reparam=False, init_by_real_act=False, **kwargs):
        if model.config.model_type != "llama":
            raise ValueError(f"Expected a Llama model, got {model.config.model_type!r}")
        if num_prefix <= 0:
            raise ValueError("num_prefix must be positive")
        if reparam:
            raise NotImplementedError("Llama prefix reparameterization is not implemented; use --no_reparam")
        initial_cache = None
        if init_by_real_act:
            embedding = model.get_input_embeddings()
            input_device = embedding.weight.device
            low_token_id = 3 if model.config.vocab_size > 3 else 0
            input_ids = torch.randint(
                low=low_token_id,
                high=model.config.vocab_size,
                size=(1, num_prefix),
                dtype=torch.long,
                device=input_device,
            )
            with torch.inference_mode():
                initial_cache = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    return_dict=True,
                ).past_key_values
            if initial_cache is None:
                raise ValueError("Llama did not return past_key_values for prefix initialization")

        for parameter in model.parameters():
            parameter.requires_grad = False
        model.prefix_encoder = LlamaPrefixEncoder(model, num_prefix)
        if initial_cache is not None:
            model.prefix_encoder.initialize_from_cache(initial_cache)
            logger.info("Initialized Llama prefix from real token KV activations")
        model.prefix_original_forward = model.forward
        model.forward = MethodType(_prefix_forward, model)
        model.prefix_original_prepare_inputs_for_generation = model.prepare_inputs_for_generation
        model.prepare_inputs_for_generation = MethodType(_prefix_prepare_inputs_for_generation, model)
        logger.info("Enabled direct Llama KV prefix tuning with %d prefix tokens", num_prefix)
