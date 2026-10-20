# opacus_patch.py
import torch
import opacus.optimizers.optimizer as opacus_opt
from opacus.optimizers.optimizer import DPOptimizer

def apply_bfloat16_patches():

    original_generate_noise = opacus_opt._generate_noise
    def safe_generate_noise(std, reference, generator=None, secure_mode=False):
        noise = original_generate_noise(std, reference, generator, secure_mode)
        return noise.to(device=reference.device, dtype=reference.dtype)
    opacus_opt._generate_noise = safe_generate_noise

    original_einsum = torch.einsum
    def safe_einsum(equation, *operands):
        if len(operands) == 2:
            target_dtype = operands[1].dtype
            target_device = operands[1].device
            safe_op0 = operands[0].to(device=target_device, dtype=target_dtype)
            safe_op1 = operands[1].to(device=target_device, dtype=target_dtype)
            return original_einsum(equation, safe_op0, safe_op1)
        return original_einsum(equation, *operands)

    original_clip_and_accumulate = DPOptimizer.clip_and_accumulate
    def safe_clip_and_accumulate(self):
        torch.einsum = safe_einsum
        try:
            original_clip_and_accumulate(self)
        finally:
            torch.einsum = original_einsum
    DPOptimizer.clip_and_accumulate = safe_clip_and_accumulate

    original_add_noise = DPOptimizer.add_noise
    def safe_add_noise(self):
        original_add_noise(self)
        for p in self.params:
            if p.grad is not None and p.grad.dtype != p.dtype:
                p.grad = p.grad.to(p.dtype)
    DPOptimizer.add_noise = safe_add_noise
    
    print("[Info] Opacus BFloat16 patches applied successfully.")