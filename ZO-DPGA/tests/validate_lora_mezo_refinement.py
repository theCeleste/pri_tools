import sys
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLAMA_DIR))

from run_formal_experiment import command_for
from run_lora_zo_dpga_stage import completion_state
from run_lora_mezo_refinement import expand_stage, load_yaml


def main():
    config = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_refinement.yaml")
    selection = load_yaml(LLAMA_DIR / "configs" / "lora_mezo_refinement_selection.yaml")
    expected = {"boundary": 12, "architecture": 7, "scale": 7, "horizon": 6, "confirm": 6}
    identities = set()
    for stage, count in expected.items():
        jobs = expand_stage(config, selection, stage)
        assert len(jobs) == count, (stage, len(jobs))
        for job in jobs:
            command = command_for(job)
            assert job["method"] == "mezo" and job["mode"] == "lora"
            assert "--trainer" in command and command[command.index("--trainer") + 1] == "zo"
            assert "--zo_dpga" not in command and "--dp_epsilon" not in command
            assert "--dev_only" in command and "--lora" in command
            assert "--lora_target_modules" in command and "--lora_num_layers" in command
            assert job["common"]["gradient_accumulation_steps"] == 1
            assert job["experiment_id"] not in identities
            identities.add(job["experiment_id"])
    print(f"PASS LoRA MeZO refinement matrix: {len(identities)} unique jobs")


if __name__ == "__main__":
    main()
