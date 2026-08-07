import sys
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for
from run_lora_mezo_exploration import expand_stage, load_yaml


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_exploration.yaml")
    selection = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_exploration_selection.yaml")
    expected = {"optimizer": 12, "structure": 8, "budget": 6, "confirm": 6}
    identities = set()
    for stage, count in expected.items():
        jobs = expand_stage(config, selection, stage)
        assert len(jobs) == count, (stage, len(jobs))
        for job in jobs:
            assert job["method"] == "mezo" and job["mode"] == "lora"
            command = command_for(job)
            assert "--trainer" in command and command[command.index("--trainer") + 1] == "zo"
            assert "--zo_dpga" not in command and "--dev_only" in command
            assert "--lora" in command and "--lora_target_modules" in command
            assert job["common"]["gradient_accumulation_steps"] == 1
            assert job["dp_epsilon"] is None
            assert job["experiment_id"] not in identities
            identities.add(job["experiment_id"])
    print(f"PASS LoRA MeZO exploration matrix: {len(identities)} unique jobs")


if __name__ == "__main__":
    main()
