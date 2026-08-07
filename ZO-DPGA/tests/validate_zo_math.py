import sys
from functools import partial
from pathlib import Path

import torch


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trainer import OurTrainer  # noqa: E402
from src.utils import forward_wrap_with_option_len_zo_dpga  # noqa: E402
from validate_checkpoint import TinyCausalDataset, build_args, build_model, collate  # noqa: E402


def build_trainer(tmp_path):
    model = build_model("lora")
    trainer = OurTrainer(
        model=model,
        args=build_args(tmp_path, max_steps=1),
        train_dataset=TinyCausalDataset(),
        data_collator=collate,
    )
    trainer.named_parameters_to_optim = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return model, trainer


def assert_perturbation_restores(model, trainer):
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.named_parameters_to_optim
    }
    trainer.zo_random_seed = 123456
    trainer.zo_perturb_parameters(scaling_factor=1)
    trainer.zo_perturb_parameters(scaling_factor=-2)
    trainer.zo_perturb_parameters(scaling_factor=1)
    for name, parameter in trainer.named_parameters_to_optim:
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=2e-7)
    print("PASS perturbation restoration")


def assert_directional_derivative(model, trainer):
    batch = collate([TinyCausalDataset()[0], TinyCausalDataset()[1]])
    model.train()
    model.zero_grad()
    loss = model(**batch).loss
    loss.backward()

    seed = 98765
    torch.manual_seed(seed)
    exact = torch.zeros((), dtype=torch.float64)
    for _, parameter in trainer.named_parameters_to_optim:
        direction = torch.normal(
            mean=0,
            std=1,
            size=parameter.shape,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        exact += (parameter.grad * direction).sum().double().cpu()

    model.zero_grad(set_to_none=True)
    trainer.zo_random_seed = seed
    trainer.zo_perturb_parameters(scaling_factor=1)
    with torch.inference_mode():
        loss_plus = model(**batch).loss
    trainer.zo_perturb_parameters(scaling_factor=-2)
    with torch.inference_mode():
        loss_minus = model(**batch).loss
    trainer.zo_perturb_parameters(scaling_factor=1)
    estimated = ((loss_plus - loss_minus) / (2 * trainer.args.zo_eps)).double().cpu()

    torch.testing.assert_close(estimated, exact, rtol=5e-2, atol=2e-3)
    print(f"PASS directional derivative: exact={exact.item():.6f} estimated={estimated.item():.6f}")


def assert_dp_clip(trainer):
    values = torch.tensor([-10.0, -1.0, -0.25, 0.0, 0.25, 1.0, 10.0])
    clipped = trainer.zo_dpga_clip(values, C=0.5)
    expected = torch.tensor([-0.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.5])
    torch.testing.assert_close(clipped, expected)
    print("PASS DP clipping")


def assert_per_example_loss():
    model = build_model("lora")
    model.original_forward = model.forward
    model.forward = partial(
        forward_wrap_with_option_len_zo_dpga.__get__(model, type(model)),
        zo_dpga=True,
    )
    dataset = TinyCausalDataset(size=3)
    batch = collate([dataset[0], dataset[1], dataset[2]])
    outputs = model(**batch, option_len=[2, 3, 4])
    if outputs.loss.shape != (3,):
        raise AssertionError(f"Expected per-example loss shape (3,), got {tuple(outputs.loss.shape)}")
    if not torch.isfinite(outputs.loss).all():
        raise AssertionError(f"Per-example loss contains non-finite values: {outputs.loss}")
    print("PASS per-example ZODPGA loss")


def main():
    tmp_path = LLAMA_DIR / "tests" / "_checkpoint_validation" / "zo_math"
    model, trainer = build_trainer(tmp_path)
    assert_perturbation_restores(model, trainer)
    assert_directional_derivative(model, trainer)
    assert_dp_clip(trainer)
    assert_per_example_loss()


if __name__ == "__main__":
    main()
