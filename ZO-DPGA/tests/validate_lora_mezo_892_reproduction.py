import sys
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import load_yaml
from run_lora_mezo_892_reproduction import build_job


def value_after(command, flag):
    return command[command.index(flag) + 1]


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_892_reproduction.yaml")
    command = command_for(build_job(config))
    expected = {
        "--seed": "0", "--train_set_seed": "0",
        "--learning_rate": "0.0002", "--zo_eps": "0.006",
        "--per_device_train_batch_size": "4", "--max_steps": "10000",
        "--lora_r": "8", "--lora_alpha": "16",
        "--lora_target_modules": "q_proj,k_proj,v_proj",
        "--lora_num_layers": "16", "--lr_scheduler_type": "constant",
        "--warmup_ratio": "0.0",
    }
    for flag, value in expected.items():
        if value_after(command, flag) != value:
            raise AssertionError(f"{flag}: expected {value}, got {value_after(command, flag)}")
    for flag in ("--dev_only", "--load_bfloat16", "--train_as_classification", "--lora"):
        if flag not in command:
            raise AssertionError(f"Missing frozen flag {flag}")
    for forbidden in ("--zo_dpga", "--dp_epsilon", "--zo_dpga_clip_threshold"):
        if forbidden in command:
            raise AssertionError(f"Unexpected ZODPGA flag {forbidden}")
    reference = config["expected_reference"]
    if reference != {
        "metric_name": "dev_accuracy", "metric_value": 0.892,
        "best_checkpoint_step": 10000, "best_eval_loss": 0.2883593738079071,
    }:
        raise AssertionError("Frozen reference metadata changed")
    print("PASS frozen 89.2% LoRA MeZO reproduction command")


if __name__ == "__main__":
    main()
