# coding=utf-8
# LLaMA model with privacy protection mechanisms for sequence classification
# Based on the privacy mechanisms from modeling_roberta_privacy.py

import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributions.laplace import Laplace

from transformers import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaConfig,
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import (
    Cache, 
    DynamicCache
)

# from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

logger = logging.get_logger(__name__)


# ============================================================================
# Custom Output Classes with Privacy Support
# ============================================================================

@dataclass
class SequenceClassifierOutputWithPrivacy(ModelOutput):
    """
    Output class for sequence classification models with privacy loss.
    
    Args:
        loss: Classification loss (optional)
        privacy_loss: Privacy protection loss from CenterLoss (optional)
        logits: Classification scores before SoftMax
        hidden_states: Hidden states from all layers (optional)
        attentions: Attention weights (optional)
    """
    loss: Optional[torch.FloatTensor] = None
    privacy_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPrivacy(ModelOutput):
    """
    Base model output with privacy-related hidden states.
    
    Args:
        last_hidden_state: Sequence of hidden states from the last layer
        hidden_states: Hidden states from all layers (optional)
        attentions: Attention weights (optional)
        past_key_values: Cached key/value pairs (optional)
        client_hidden_states: Clean hidden states before noise addition (optional)
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    client_hidden_states: Optional[torch.FloatTensor] = None


# ============================================================================
# Privacy Protection Components
# ============================================================================

class CenterLoss(nn.Module):
    """
    Center loss for privacy protection through clustering.
    Maintains cluster centers and computes losses to:
    1. Pull samples toward their cluster centers
    2. Push different cluster centers apart
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize cluster embeddings
        self.cluster_embedding = nn.Embedding(
            config.cluster_num, 
            config.hidden_size
        )
        self.loss_fct = nn.MSELoss()
    
    def forward(self, clean_hidden_states, cluster_ids):
        """
        Compute center loss and cluster separation loss.
        
        Args:
            clean_hidden_states: (batch_size * seq_len, hidden_size) - flattened hidden states
            cluster_ids: (batch_size * seq_len,) - cluster assignments
        
        Returns:
            Dictionary with center_loss and cluster_away_loss
        """
        # Get target cluster centers
        target_cluster = self.cluster_embedding(cluster_ids)
        
        # Center loss: pull samples to their cluster centers
        center_loss = self.loss_fct(clean_hidden_states, target_cluster)
        
        # Cluster separation loss: push cluster centers apart
        cluster_centers = self.cluster_embedding.weight
        cluster_num = cluster_centers.shape[0]
        
        # Compute pairwise distances between cluster centers
        cluster_distances = torch.cdist(cluster_centers, cluster_centers)
        # Only consider upper triangular part (avoid double counting)
        cluster_away_loss = -torch.triu(cluster_distances, diagonal=1).sum() / ((1 + cluster_num) * cluster_num / 2)
        
        return {
            'center_loss': self.config.w_cluster_close * center_loss,
            'cluster_away_loss': self.config.w_cluster_away * cluster_away_loss
        }


class LlamaClassificationHead(nn.Module):
    """
    Classification head for LLaMA sequence classification.
    Similar to RobertaClassificationHead but adapted for LLaMA.
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None
            else config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob')
            else 0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features, **kwargs):
        """
        Args:
            features: (batch_size, seq_len, hidden_size)
        Returns:
            logits: (batch_size, num_labels)
        """
        # For LLaMA, we typically use the last token for classification
        # instead of the first token like in BERT/RoBERTa
        x = features[:, -1, :]  # Take the last token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ============================================================================
# Modified LLaMA Model with Privacy Support
# ============================================================================

# class LlamaModelWithPrivacy(LlamaModel):
#     """
#     LLaMA Model with privacy protection through noise injection.
#     """
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config

#         # Laplace noise generator (initialized once)
#         if hasattr(config, "eps") and config.eps > 0:
#             self.m = Laplace(
#                 loc=torch.tensor(0.0, device='cpu', dtype=torch.float32),
#                 scale=torch.tensor(1.0 / config.eps, device='cpu', dtype=torch.float32)
#             )
#         else:
#             self.m = None

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cluster_ids: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPrivacy]:
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         # Retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")
        
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
        
#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
        
#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()
        
#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)
        
#         # Prepare attention mask
#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
#         )
        
#         hidden_states = inputs_embeds
        
#         # Decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
        
#         client_hidden_states = None
        
#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)
            
#             past_key_value = past_key_values[idx] if past_key_values is not None else None
            
#             # Apply decoder layer
#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#             )
            
#             hidden_states = layer_outputs[0]
            
#             # Add noise at target layer for privacy protection
#             if idx == self.config.target_layer - 1:
#                 # Save clean hidden states before adding noise
#                 client_hidden_states = hidden_states
                
#                 # Generate and add Laplace noise
#                 noise = self.noise_generator.sample(hidden_states.shape).to(hidden_states.device, dtype=hidden_states.dtype)
#                 hidden_states = hidden_states + noise.detach() * self.config.w_noise  # detach to prevent gradient through noise
            
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)
        
#         hidden_states = self.norm(hidden_states)
        
#         # Add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)
        
#         next_cache = next_decoder_cache if use_cache else None
        
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
#         return BaseModelOutputWithPrivacy(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#             client_hidden_states=client_hidden_states
#         )

# def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
#     """
#     Prepare a 4D causal attention mask for LLaMA.
#     Returns a mask of shape [batch_size, 1, seq_length, seq_length + past_key_values_length].
#     """
#     batch_size, seq_length = input_shape
#     device = inputs_embeds.device
#     dtype = inputs_embeds.dtype

#     if attention_mask is None:
#         attention_mask = torch.ones((batch_size, seq_length), device=device)

#     # [batch, 1, 1, seq_length]
#     extended_attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)

#     # [seq_length, seq_length + past_key_values_length]
#     causal_mask = torch.tril(torch.ones(
#         (seq_length, seq_length + past_key_values_length),
#         device=device,
#         dtype=dtype
#     ))

#     extended_attention_mask = extended_attention_mask * causal_mask
#     return extended_attention_mask

class LlamaModelWithPrivacy(LlamaModel):
    """
    LLaMA Model with privacy protection via Laplace noise injection.
    Noise is added after rotary embeddings at a specified decoder layer.
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,  # 保留你之前参数
    ) -> BaseModelOutputWithPrivacy:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, hidden_size = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 构建 causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        # print("hidden_states:",hidden_states.shape)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # print("position_embeddings:", position_embeddings[0].shape, position_embeddings[1].shape)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        client_hidden_states = None

        # --- decoder 层循环 ---
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # forward decoder layer，并传入 rotary embedding
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs
            # print("hidden_states:",hidden_states.shape)

            # 在目标层加噪声
            if idx == getattr(self.config, "target_layer", -1):
                # print("target_layer:",idx)
                client_hidden_states = hidden_states
                laplace = Laplace(
                    loc=torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
                    scale=torch.tensor(1.0 / getattr(self.config, "eps", 1.0),
                                       device=hidden_states.device, dtype=hidden_states.dtype)
                )
                noise = laplace.sample(hidden_states.shape).to(hidden_states.device, hidden_states.dtype)
                # print("pre:",hidden_states.shape)
                hidden_states = hidden_states + getattr(self.config, "w_noise", 0.1) * noise
                # print("post:",hidden_states.shape)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPrivacy(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            past_key_values=past_key_values,
            client_hidden_states=client_hidden_states,
        )



# ============================================================================
# Main LLaMA Sequence Classification Model
# ============================================================================

@add_start_docstrings(
    """
    LLaMA Model transformer with a sequence classification head on top (a linear layer on top of
    the pooled output) with privacy protection mechanisms.
    """,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    LLaMA for sequence classification with privacy protection.
    Includes differential privacy noise and clustering-based privacy mechanisms.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Initialize LLaMA model with privacy support
        self.model = LlamaModelWithPrivacy(config)
        
        # Classification head
        self.classifier = LlamaClassificationHead(config)
        
        # Privacy loss module
        self.privacy_loss = CenterLoss(config)
        
        # Initialize weights and apply final processing
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPrivacy]:
        """
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for caching
            inputs_embeds: Input embeddings
            labels: Labels for classification loss
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput
            cluster_ids: Cluster assignments for privacy protection
        
        Returns:
            SequenceClassifierOutputWithPrivacy containing loss, logits, and optional privacy loss
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through LLaMA model
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cluster_ids=cluster_ids,
        )
        
        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        
        loss = None
        privacy_loss = None
        
        # Compute classification loss
        if labels is not None:
            # Determine problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # Compute privacy loss if cluster_ids are provided
        if cluster_ids is not None and transformer_outputs.client_hidden_states is not None:
            # Filter valid tokens (non-padding, non-special tokens)
            # For LLaMA, we might use different special token IDs
            pad_token_id = self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0
            eos_token_id = self.config.eos_token_id if hasattr(self.config, 'eos_token_id') else 2
            
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
            else:
                valid_mask = (input_ids != pad_token_id) & (input_ids != eos_token_id)
            
            # Flatten the tensors for processing
            batch_size, seq_len = input_ids.shape if input_ids is not None else valid_mask.shape
            client_hidden_states = transformer_outputs.client_hidden_states
            
            # Reshape and filter
            client_hidden_states_flat = client_hidden_states.view(-1, client_hidden_states.size(-1))
            cluster_ids_flat = cluster_ids.view(-1)
            valid_mask_flat = valid_mask.view(-1)
            
            # Apply mask
            valid_hidden_states = client_hidden_states_flat[valid_mask_flat]
            valid_cluster_ids = cluster_ids_flat[valid_mask_flat]
            
            # Compute privacy losses
            privacy_loss = self.privacy_loss(valid_hidden_states, valid_cluster_ids)
        
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPrivacy(
            loss=loss,
            privacy_loss=privacy_loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# ============================================================================
# Configuration Extensions
# ============================================================================

def add_privacy_config(config):
    """
    Add privacy-related configuration parameters to LlamaConfig.
    
    Args:
        config: LlamaConfig instance
    
    Returns:
        Modified config with privacy parameters
    """
    # Privacy parameters with defaults
    privacy_params = {
        'cluster_num': 10,           # Number of clusters
        'target_layer': 16,          # Layer to add noise (for LLaMA-7B, typically has 32 layers)
        'eps': 1.0,                  # Differential privacy epsilon
        'w_noise': 0.1,              # Noise weight
        'w_cluster_close': 1.0,      # Weight for cluster attraction loss
        'w_cluster_away': 0.5,       # Weight for cluster separation loss
        'target_break': False,       # Whether to stop after target layer
        'classifier_dropout': 0.1,   # Dropout for classification head
    }
    
    for key, default_value in privacy_params.items():
        if not hasattr(config, key):
            setattr(config, key, default_value)
    
    return config
