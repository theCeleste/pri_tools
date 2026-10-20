"""Small BF16 compatibility patches used by the existing ChatGLM/Opacus work."""


def apply_bfloat16_patches() -> None:
    import torch
    import opacus.optimizers.optimizer as opacus_optimizer
    from opacus.optimizers.optimizer import DPOptimizer

    original_generate_noise = opacus_optimizer._generate_noise

    def generate_noise(std, reference, generator=None, secure_mode=False):
        noise = original_generate_noise(std, reference, generator, secure_mode)
        return noise.to(device=reference.device, dtype=reference.dtype)

    opacus_optimizer._generate_noise = generate_noise
    original_einsum = torch.einsum

    def safe_einsum(equation, *operands):
        if len(operands) == 2:
            target = operands[1]
            operands = tuple(value.to(device=target.device, dtype=target.dtype) for value in operands)
        return original_einsum(equation, *operands)

    original_clip = DPOptimizer.clip_and_accumulate

    def clip_and_accumulate(optimizer):
        torch.einsum = safe_einsum
        try:
            return original_clip(optimizer)
        finally:
            torch.einsum = original_einsum

    DPOptimizer.clip_and_accumulate = clip_and_accumulate
