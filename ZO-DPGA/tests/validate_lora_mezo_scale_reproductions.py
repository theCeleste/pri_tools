import sys
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for
from run_lora_zo_dpga_matrix import load_yaml
from run_lora_mezo_scale_reproduction import build_job


def value_after(command, flag):
    return command[command.index(flag) + 1]


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_scale_reproductions.yaml")
    expected_variants = {
        "r8_a16_892": ("8", "16", 0.892, 10000),
        "r16_a32_900": ("16", "32", 0.900, 8000),
    }
    for name, (rank, alpha, metric, best_step) in expected_variants.items():
        job = build_job(config, name)
        command = command_for(job)
        expected_flags = {
            "--seed": "0", "--train_set_seed": "0",
            "--learning_rate": "0.0002", "--zo_eps": "0.006",
            "--per_device_train_batch_size": "4", "--max_steps": "10000",
            "--lora_r": rank, "--lora_alpha": alpha,
            "--lora_target_modules": "q_proj,k_proj,v_proj",
            "--lora_num_layers": "16", "--lr_scheduler_type": "constant",
            "--warmup_ratio": "0.0",
        }
        for flag, value in expected_flags.items():
            if value_after(command, flag) != value:
                raise AssertionError(f"{name} {flag}: expected {value}, got {value_after(command, flag)}")
        for flag in ("--dev_only", "--load_bfloat16", "--train_as_classification", "--lora"):
            if flag not in command:
                raise AssertionError(f"{name}: missing {flag}")
        for forbidden in ("--zo_dpga", "--dp_epsilon", "--zo_dpga_clip_threshold"):
            if forbidden in command:
                raise AssertionError(f"{name}: unexpected {forbidden}")
        reference = config["variants"][name]["expected_reference"]
        if reference["metric_value"] != metric or reference["best_checkpoint_step"] != best_step:
            raise AssertionError(f"{name}: reference metadata changed")
    print("PASS frozen LoRA MeZO scale reproduction commands")


if __name__ == "__main__":
    main()
