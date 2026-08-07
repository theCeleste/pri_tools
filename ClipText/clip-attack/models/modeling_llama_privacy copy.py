# coding=utf-8
"""LLaMA model with privacy-aware sequence classification head.

本实现与现有的 Roberta 隐私版本保持接口一致：
- 向前传递 cluster_ids，用于 CenterLoss 计算隐私损失；
- 在输出中附带 privacy_loss；
- 通过 output_hidden_states 拿到 target_layer 的隐藏态作为 client_hidden_states；
- 可在最终层（当 target_layer 等于最后一层）对输出注入拉普拉斯噪声（受 config.w_noise 控制）。

说明：为了最小侵入改造，这里不修改 LLaMA 的层级内部执行，仅在需要时对最终层输出加噪；
若需在中间层加噪并继续前向，需要自定义解码层循环，工程侵入较大，可在后续版本迭代。
"""

from typing import Optional, Tuple, Union, List
import logging

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributions.laplace import Laplace

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPast
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

# 复用已有的 CenterLoss 与自定义带 privacy_loss 的输出结构
from .modeling_roberta_privacy import CenterLoss  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class BaseModelOutputWithPastAndClientHiddenStates(BaseModelOutputWithPast):
    """
    扩展BaseModelOutputWithPast以支持client_hidden_states
    """
    client_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class SequenceClassifierOutputWithPrivacy(SequenceClassifierOutput):
    """
    扩展SequenceClassifierOutput以支持privacy_loss和client_hidden_states
    """
    privacy_loss: Optional[torch.FloatTensor] = None
    client_hidden_states: Optional[torch.FloatTensor] = None


class LlamaModelPrivacy(LlamaModel):
    """
    自定义的LlamaModel，支持隐私保护机制
    参考RoBERTa版本的实现，在特定层进行client_hidden_states保存和噪声注入
    """
    
    def __init__(self, config):
        super().__init__(config)
        # 初始化拉普拉斯分布用于噪声注入
        self.m = Laplace(torch.tensor(0.0), torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 使用父类的forward方法，但需要重写decoder层循环来添加隐私保护
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or inputs_embeds")

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers with privacy protection
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # 用于保存client_hidden_states
        client_hidden_states = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            # 在目标层进行隐私保护处理 - 参考RoBERTa版本
            if idx == (self.config.target_layer - 1):
                target_noise = self.m.sample(hidden_states.shape).type_as(hidden_states) * self.config.w_noise
                client_hidden_states = hidden_states
                hidden_states = hidden_states + target_noise
                
                if self.config.target_break:
                    break

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPastAndClientHiddenStates(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            client_hidden_states=client_hidden_states,
        )


class LlamaClassificationHead(nn.Module):
    """与 RobertaClassificationHead 等价的简单分类头。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 兼容不同配置字段：优先 classifier_dropout，其次 hidden_dropout_prob，再次 hidden_dropout，最后兜底 0.0
        _cd = getattr(config, "classifier_dropout", None)
        if _cd is None:
            _cd = getattr(config, "hidden_dropout_prob", None)
        if _cd is None:
            _cd = getattr(config, "hidden_dropout", 0.0)
        self.dropout = nn.Dropout(_cd)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 对于LLaMA decoder-only模型，使用所有token的平均值进行分类
        # 这相当于RoBERTa中[CLS] token的作用
        x = features.mean(dim=1)  # 对所有token取平均
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LlamaForSequenceClassificationPrivacy(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = LlamaModel(config)  # 使用原始的LlamaModel
        self.classifier = LlamaClassificationHead(config)

        self.privacy_loss = CenterLoss(config)

        # Initialize weights and apply final processing
        self.post_init()

        # 懒初始化噪声分布，按需在 forward 内部绑定到正确设备/精度
        self._laplace_dist = None

    def _get_laplace(self, device, dtype):
        if self._laplace_dist is None or self._laplace_dist.loc.device != device or self._laplace_dist.loc.dtype != dtype:
            scale = 1.0 / max(1e-6, float(getattr(self.config, "eps", 1.0)))
            self._laplace_dist = Laplace(
                loc=torch.tensor(0.0, device=device, dtype=dtype),
                scale=torch.tensor(scale, device=device, dtype=dtype),
            )
        return self._laplace_dist

    @add_start_docstrings_to_model_forward("LLaMA privacy-aware sequence classification forward")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cluster_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 确保输出隐藏态以获取 target_layer 的 client_hidden_states
        if output_hidden_states is None:
            output_hidden_states = True

        # 与RoBERTa完全一致的噪声注入机制
        client_hidden_states = None
        target_layer = int(getattr(self.config, "target_layer", 3))
        
        # 调用原始LlamaModel
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # 使用hook机制在目标层注入噪声，与RoBERTa保持一致的隐私保护
        client_hidden_states = None
        target_layer = int(getattr(self.config, "target_layer", 3))
        
        def hook_fn(module, input, output):
            nonlocal client_hidden_states
            # LlamaDecoderLayer的输出是tuple，第一个元素是hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # 确保hidden_states是3D的 [batch_size, seq_len, hidden_size]
            if len(hidden_states.shape) != 3:
                return
            
            # 保存注入噪声前的状态作为client_hidden_states
            client_hidden_states = hidden_states.clone()
            
            # 在目标层注入噪声（与RoBERTa完全一致）
            if getattr(self.config, "w_noise", 0):
                eps = getattr(self.config, "eps", 1.0)
                m = torch.distributions.Laplace(
                    loc=torch.tensor(0, device=hidden_states.device, dtype=torch.float),
                    scale=torch.tensor(1/eps, device=hidden_states.device, dtype=torch.float)
                )
                target_noise = m.sample(hidden_states.shape).type_as(hidden_states) * self.config.w_noise
                
                # 直接修改tensor的内容，这样会影响后续层
                hidden_states.add_(target_noise)
        
        # 注册hook到目标层
        hooks = []
        for i, layer in enumerate(self.model.layers):
            if i == (target_layer - 1):
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)
                break  # 只注册目标层
        
        try:
            # 调用原始LlamaModel
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()
        
        
        sequence_output = outputs.last_hidden_state

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
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

        # 计算隐私损失 - 与RoBERTa版本完全一致
        privacy_loss = None
        if cluster_ids is not None and client_hidden_states is not None:
            # 检查形状是否匹配
            if len(client_hidden_states.shape) != 3:
                print(f"ERROR: client_hidden_states shape mismatch: {client_hidden_states.shape}")
                print(f"Expected 3D tensor [batch_size, seq_len, hidden_size]")
                return SequenceClassifierOutputWithPrivacy(
                    loss=loss,
                    privacy_loss=None,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    client_hidden_states=client_hidden_states,
                )
            
            # 使用与RoBERTa相同的valid_ids过滤条件
            valid_ids = (attention_mask == 1) & (input_ids != 2) & (input_ids != 0)
            
            # 展平到 token 粒度（与RoBERTa完全一致）
            chs = client_hidden_states[valid_ids]
            cids = cluster_ids[valid_ids]
            
            if len(chs) > 0 and len(cids) > 0:  # 确保有有效的token
                privacy_loss = self.privacy_loss(chs, cids)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        # 与RoBERTa版本保持完全一致的输出格式
        return SequenceClassifierOutputWithPrivacy(
            loss=loss,
            privacy_loss=privacy_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            client_hidden_states=client_hidden_states,
        )



