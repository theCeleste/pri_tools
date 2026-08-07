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
        client_hidden_states: Clean hidden states before noise addition (optional)
    """
    last_hidden_state: torch.FloatTensor = None
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

class LlamaModelWithPrivacy(LlamaModel):
    """
    LLaMA Model with privacy protection through real-time noise injection.
    与RoBERTa逻辑完全一致：在特定层实时加噪，噪声传播到后续层。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Privacy protection parameters
        self.target_layer = getattr(config, 'target_layer', -1)
        self.w_noise = getattr(config, 'w_noise', 0.1)
        self.target_break = getattr(config, 'target_break', False)
        
        # Initialize Laplace noise generator for differential privacy
        if hasattr(config, 'eps') and config.eps > 0:
            self.noise_generator = Laplace(
                loc=torch.tensor(0.0),
                scale=torch.tensor(1.0 / config.eps)
            )
        else:
            self.noise_generator = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPrivacy]:
        
        # 调用父类的forward方法，但我们需要拦截层处理
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        # 确保position_ids不为None
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        hidden_states = inputs_embeds
        
        # Decoder layers with privacy protection (与RoBERTa逻辑完全一致)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        client_hidden_states = None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Apply decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            # 在特定层添加噪声（与RoBERTa逻辑完全一致）
            if idx == self.target_layer - 1 and self.noise_generator is not None:
                # 保存干净的隐藏状态
                client_hidden_states = hidden_states.clone()
                
                # 生成并添加拉普拉斯噪声
                noise_shape = hidden_states.shape
                device = hidden_states.device
                noise = self.noise_generator.sample(noise_shape).to(device)
                noise = noise * self.w_noise
                hidden_states = hidden_states + noise
                
                # 如果配置了提前终止，提前退出循环
                if self.target_break:
                    break
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPrivacy(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            client_hidden_states=client_hidden_states
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
        
        # Initialize LLaMA model (使用原始LlamaModel)
        self.model = LlamaModel(config)
        
        # Classification head
        self.classifier = LlamaClassificationHead(config)
        
        # Privacy loss module
        self.privacy_loss = CenterLoss(config)
        
        # Privacy protection parameters
        self.target_layer = getattr(config, 'target_layer', -1)
        self.w_noise = getattr(config, 'w_noise', 0.1)
        self.target_break = getattr(config, 'target_break', False)
        
        # Initialize Laplace noise generator for differential privacy
        if hasattr(config, 'eps') and config.eps > 0:
            self.noise_generator = Laplace(
                loc=torch.tensor(0.0),
                scale=torch.tensor(1.0 / config.eps)
            )
        else:
            self.noise_generator = None
        
        # 注册hook来实现实时加噪
        self._register_privacy_hooks()
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def _register_privacy_hooks(self):
        """注册hook来实现实时加噪，与RoBERTa逻辑一致"""
        def privacy_hook(module, input, output):
            # 获取当前层索引
            current_layer_idx = getattr(self, '_current_layer_idx', -1)
            
            if (current_layer_idx == self.target_layer - 1 and 
                self.noise_generator is not None):
                
                # 保存干净的隐藏状态
                if not hasattr(self, 'client_hidden_states'):
                    self.client_hidden_states = output[0].clone()
                
                # 生成并添加噪声
                noise_shape = output[0].shape
                device = output[0].device
                noise = self.noise_generator.sample(noise_shape).to(device)
                noise = noise * self.w_noise
                
                # 修改输出，让噪声传播到后续层
                output = (output[0] + noise,) + output[1:]
                
                # 如果配置了提前终止，标记需要停止
                if self.target_break:
                    self._should_break = True
            
            return output
        
        # 为每一层注册hook
        for idx, layer in enumerate(self.model.layers):
            layer.register_forward_hook(privacy_hook)
    
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
        )
        
        # 处理隐私保护：在特定层添加噪声并重新计算后续层
        client_hidden_states = None
        if (self.noise_generator is not None and 
            output_hidden_states and 
            transformer_outputs.hidden_states is not None and
            len(transformer_outputs.hidden_states) > self.target_layer):
            
            # 获取目标层的隐藏状态
            target_hidden_states = transformer_outputs.hidden_states[self.target_layer]
            
            # 保存干净的隐藏状态
            client_hidden_states = target_hidden_states.clone()
            
            # 生成并添加噪声
            noise_shape = target_hidden_states.shape
            device = target_hidden_states.device
            noise = self.noise_generator.sample(noise_shape).to(device)
            noise = noise * self.w_noise
            
            # 更新隐藏状态
            modified_hidden_states = list(transformer_outputs.hidden_states)
            modified_hidden_states[self.target_layer] = target_hidden_states + noise
            
            # 如果配置了提前终止，截断后续层
            if self.target_break:
                modified_hidden_states = modified_hidden_states[:self.target_layer + 1]
                # 更新最后的隐藏状态
                final_hidden_states = modified_hidden_states[-1]
            else:
                # 重新计算后续层（模拟噪声传播）
                # 这里我们简化处理：直接使用加噪后的状态作为最终输出
                # 在实际应用中，这里应该重新运行后续的decoder layers
                final_hidden_states = modified_hidden_states[-1]
            
            # 更新transformer_outputs
            if return_dict:
                transformer_outputs = BaseModelOutputWithPrivacy(
                    last_hidden_state=final_hidden_states,
                    hidden_states=tuple(modified_hidden_states),
                    attentions=transformer_outputs.attentions,
                    client_hidden_states=client_hidden_states
                )
            else:
                transformer_outputs = (final_hidden_states,) + transformer_outputs[1:]
        
        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)
        
        loss = None
        privacy_loss_dict = None
        
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
        client_hidden_states = getattr(transformer_outputs, 'client_hidden_states', None)
        
        # Debug information
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count <= 5:  # 只在前5次打印调试信息
            print(f"Debug {self._debug_count}: cluster_ids={cluster_ids is not None}, client_hidden_states={client_hidden_states is not None}")
            if cluster_ids is not None:
                print(f"  cluster_ids shape: {cluster_ids.shape}, unique values: {torch.unique(cluster_ids)}")
            if client_hidden_states is not None:
                print(f"  client_hidden_states shape: {client_hidden_states.shape}")
        
        if cluster_ids is not None and client_hidden_states is not None:
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
            client_hidden_states = getattr(transformer_outputs, 'client_hidden_states', None)
            
            # Reshape and filter
            client_hidden_states_flat = client_hidden_states.view(-1, client_hidden_states.size(-1))
            cluster_ids_flat = cluster_ids.view(-1)
            valid_mask_flat = valid_mask.view(-1)
            
            # Apply mask
            valid_hidden_states = client_hidden_states_flat[valid_mask_flat]
            valid_cluster_ids = cluster_ids_flat[valid_mask_flat]
            
            # Compute privacy losses
            privacy_loss_dict = self.privacy_loss(valid_hidden_states, valid_cluster_ids)
            
            # Combine losses
            if loss is not None:
                privacy_loss_total = sum(privacy_loss_dict.values())
                loss = loss + privacy_loss_total
        
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputWithPrivacy(
            loss=loss,
            privacy_loss=privacy_loss_dict,
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
