import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False, # Not sure if this will affect saving/loading models so just set it to be False
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRA:

    def __init__(self, model, r, alpha, float16, target_modules="q_proj,v_proj", num_layers=-1):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        if model.config.model_type != "llama":
            raise ValueError(f"Llama LoRA adapter received model_type={model.config.model_type!r}")
        target_modules = tuple(part.strip() for part in target_modules.split(",") if part.strip())
        allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        unknown_targets = set(target_modules) - allowed_targets
        if not target_modules or unknown_targets:
            raise ValueError(
                f"LoRA target_modules must be a non-empty subset of {sorted(allowed_targets)}; "
                f"unknown={sorted(unknown_targets)}"
            )
        layer_count = len(model.model.layers)
        if num_layers == -1:
            first_layer = 0
        elif 1 <= num_layers <= layer_count:
            first_layer = layer_count - num_layers
        else:
            raise ValueError(f"lora_num_layers must be -1 or between 1 and {layer_count}")
        attention_name = "self_attn"

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                parts = key.split(".")
                try:
                    layer_index = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                if layer_index < first_layer:
                    continue
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                for projection_name in target_modules:
                    original = getattr(attn, projection_name)
                    replacement = LoRALinear(
                        original.in_features,
                        original.out_features,
                        r=r,
                        lora_alpha=alpha,
                        bias=original.bias is not None,
                    ).to(device=original.weight.device, dtype=original.weight.dtype)
                    replacement.weight.data.copy_(original.weight.data)
                    if original.bias is not None:
                        replacement.bias.data.copy_(original.bias.data)
                    setattr(attn, projection_name, replacement)
        
        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
            else:
                print(n)
