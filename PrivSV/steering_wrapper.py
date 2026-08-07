import torch
from contextlib import contextmanager


class FrozenSteering:
    def __init__(self, layer_activations: dict[int, torch.Tensor]):
        self.layer_activations = layer_activations

    @contextmanager
    def apply(self, model, multiplier: float = 1.0, min_token_index: int = 0):
        hooks = []
        device = next(model.parameters()).device

        for layer_idx, vec in self.layer_activations.items():
            vec = vec.to(device)

            def hook_fn(module, inputs, output, v=vec):
                # HuggingFace decoder blocks often return a tuple
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states[:, min_token_index:, :] += multiplier * v
                    return (hidden_states,) + output[1:]
                else:
                    # In case the block directly returns a tensor
                    output[:, min_token_index:, :] += multiplier * v
                    return output


            h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()
