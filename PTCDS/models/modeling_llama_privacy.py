# coding=utf-8
"""
LLaMA model with privacy protection mechanisms for sequence classification
Compatible with transformers==4.39.3
Core idea:
- DO NOT override LlamaModel.forward (avoid rotary/cache/mask API mismatch)
- Use a forward hook on a chosen decoder layer to:
  1) save clean hidden states (before noise)
  2) add Laplace noise to the layer output
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributions.laplace import Laplace

from transformers import LlamaPreTrainedModel, LlamaModel, LlamaConfig
from transformers.modeling_outputs import ModelOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

logger = logging.get_logger(__name__)


# ============================================================================
# Output Classes
# ============================================================================

@dataclass
class SequenceClassifierOutputWithPrivacy(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    privacy_loss: Optional[Dict[str, torch.FloatTensor]] = None  # dict like {'center_loss':..., 'cluster_away_loss':...}
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# ============================================================================
# Privacy Loss (CenterLoss)
# ============================================================================

class CenterLoss(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.cluster_embedding = nn.Embedding(config.cluster_num, config.hidden_size)
        self.loss_fct = nn.MSELoss()

    def forward(self, clean_hidden_states: torch.Tensor, cluster_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # clean_hidden_states: (N, hidden), cluster_ids: (N,)
        target_cluster = self.cluster_embedding(cluster_ids)
        center_loss = self.loss_fct(clean_hidden_states, target_cluster)

        cluster_centers = self.cluster_embedding.weight  # (C, hidden)
        cluster_num = cluster_centers.shape[0]
        cluster_distances = torch.cdist(cluster_centers, cluster_centers)

        cluster_away_loss = -torch.triu(cluster_distances, diagonal=1).sum() / ((1 + cluster_num) * cluster_num / 2)

        return {
            "center_loss": float(self.config.w_cluster_close) * center_loss,
            "cluster_away_loss": float(self.config.w_cluster_away) * cluster_away_loss,
        }


# ============================================================================
# Classification Head
# ============================================================================

class LlamaClassificationHead(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (bs, seq, hidden) -> use last token
        x = features[:, -1, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


# ============================================================================
# LlamaModelWithPrivacy (hook-based, NO forward override)
# ============================================================================

class LlamaModelWithPrivacy(LlamaModel):
    """
    In transformers==4.39.3, LlamaModel does NOT expose rotary_emb at model level.
    So we must NOT rewrite forward().

    We inject noise via forward hook on a chosen decoder layer output.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # store clean hidden states at target layer output (before noise)
        self._client_hidden_states: Optional[torch.Tensor] = None

        t = getattr(config, "target_layer", None)
        if t is None:
            logger.warning("config.target_layer is None, no privacy noise will be applied.")
            return

        # target_layer is treated as 1-based in your training script
        layer_idx = int(t) - 1
        if not (0 <= layer_idx < len(self.layers)):
            raise ValueError(f"target_layer out of range: {t}, layers={len(self.layers)}")

        def _noise_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
            """
            output is usually a tuple where output[0] is hidden_states: (bs, seq, hidden)
            """
            if isinstance(output, (tuple, list)):
                hs = output[0]
                rest = tuple(output[1:])
            else:
                hs = output
                rest = None

            # save clean hidden states
            self._client_hidden_states = hs

            eps = float(getattr(self.config, "eps", 1.0))
            w_noise = float(getattr(self.config, "w_noise", 0.1))

            lap = Laplace(
                loc=torch.tensor(0.0, device=hs.device, dtype=hs.dtype),
                scale=torch.tensor(1.0 / max(eps, 1e-6), device=hs.device, dtype=hs.dtype),
            )
            noise = lap.sample(hs.shape).to(hs.device, hs.dtype)
            hs_noisy = hs + w_noise * noise.detach()

            if rest is None:
                return hs_noisy
            return (hs_noisy,) + rest

        self.layers[layer_idx].register_forward_hook(_noise_hook)
        logger.info(f"Privacy noise hook registered on decoder layer index={layer_idx} (target_layer={t})")


# ============================================================================
# Main Sequence Classification Model
# ============================================================================

@add_start_docstrings(
    "LLaMA transformer with a sequence classification head on top, plus privacy protection.",
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = LlamaModelWithPrivacy(config)
        self.classifier = LlamaClassificationHead(config)
        self.privacy_loss = CenterLoss(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward("LLaMA inputs")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,  # for privacy loss only (DO NOT pass into base llama)
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPrivacy]:

        # IMPORTANT: your training script may pass output_hidden_states=True via batch dict
        # also may pass extra kwargs; we ignore them safely
        return_dict = True if return_dict is None else bool(return_dict)

        # reset saved clean hs each forward
        self.model._client_hidden_states = None

        # call base llama (no cluster_ids here)
        base_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # keep stable
        )

        hidden_states = base_out.last_hidden_state  # (bs, seq, hidden)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # infer problem type if not set
            problem_type = getattr(self.config, "problem_type", None)
            if problem_type is None:
                if self.num_labels == 1:
                    problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int64, torch.int32):
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"

            if problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # privacy loss: use clean hidden states saved by hook
        privacy_loss = None
        clean_hs = getattr(self.model, "_client_hidden_states", None)

        if cluster_ids is not None and clean_hs is not None:
            # clean_hs: (bs, seq, hidden), cluster_ids: (bs, seq)
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
            else:
                pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
                valid_mask = (input_ids != pad_id)

            clean_flat = clean_hs.reshape(-1, clean_hs.size(-1))
            cluster_flat = cluster_ids.reshape(-1)
            mask_flat = valid_mask.reshape(-1)

            valid_hidden = clean_flat[mask_flat]
            valid_cluster = cluster_flat[mask_flat]

            privacy_loss = self.privacy_loss(valid_hidden, valid_cluster)

        if not return_dict:
            # keep compatibility if someone calls with return_dict=False
            out = (logits,)
            if output_hidden_states:
                out = out + (base_out.hidden_states,)
            if output_attentions:
                out = out + (base_out.attentions,)
            if loss is not None:
                out = (loss,) + out
            return out

        return SequenceClassifierOutputWithPrivacy(
            loss=loss,
            privacy_loss=privacy_loss,
            logits=logits,
            hidden_states=base_out.hidden_states if output_hidden_states else None,
            attentions=base_out.attentions if output_attentions else None,
        )


# ============================================================================
# Config extension
# ============================================================================

def add_privacy_config(config: LlamaConfig) -> LlamaConfig:
    privacy_params = {
        "cluster_num": 10,
        "target_layer": 16,     # 1-based
        "eps": 1.0,
        "w_noise": 0.1,
        "w_cluster_close": 1.0,
        "w_cluster_away": 0.5,
        "target_break": False,
        "classifier_dropout": 0.1,
    }
    for k, v in privacy_params.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    return config