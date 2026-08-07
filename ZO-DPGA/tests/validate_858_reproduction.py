import sys
from pathlib import Path


LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for  # noqa: E402
from run_zo_dpga_858_reproduction import build_job  # noqa: E402
from run_lora_zo_dpga_matrix import load_yaml  # noqa: E402


def value_after(command, flag):
    return command[command.index(flag) + 1]


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "zo_dpga_858_reproduction.yaml")
    job = build_job(config)
    command = command_for(job)
    expected = {
        "--seed": "2",
        "--train_set_seed": "2",
        "--learning_rate": "0.0001",
        "--zo_eps": "0.006",
        "--per_device_train_batch_size": "4",
        "--max_steps": "10000",
        "--dp_epsilon": "6.0",
        "--zo_dpga_clip_threshold": "10.0",
        "--lora_r": "8",
        "--lora_alpha": "16",
        "--lora_target_modules": "q_proj,v_proj",
        "--lora_num_layers": "16",
        "--lr_scheduler_type": "constant",
        "--warmup_ratio": "0.0",
    }
    for flag, value in expected.items():
        if value_after(command, flag) != value:
            raise AssertionError(f"{flag}: expected {value}, got {value_after(command, flag)}")
    for flag in ("--zo_dpga", "--dev_only", "--load_bfloat16", "--train_as_classification"):
        if flag not in command:
            raise AssertionError(f"Missing frozen flag {flag}")
    print("PASS frozen 85.8% reproduction command")


if __name__ == "__main__":
    main()
