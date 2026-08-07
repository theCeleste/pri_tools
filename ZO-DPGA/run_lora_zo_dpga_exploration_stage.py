import sys

from run_lora_zo_dpga_exploration import DEFAULT_CONFIG, DEFAULT_SELECTION
from run_lora_zo_dpga_stage import main


if __name__ == "__main__":
    defaults = [
        "--runner-module", "run_lora_zo_dpga_exploration",
        "--config", str(DEFAULT_CONFIG),
        "--selection", str(DEFAULT_SELECTION),
    ]
    sys.argv[1:1] = defaults
    main()
