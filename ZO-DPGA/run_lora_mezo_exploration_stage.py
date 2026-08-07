import sys

from run_lora_zo_dpga_stage import main
from run_lora_mezo_exploration import DEFAULT_CONFIG, DEFAULT_SELECTION


if __name__ == "__main__":
    defaults = [
        "--runner-module", "run_lora_mezo_exploration",
        "--config", str(DEFAULT_CONFIG),
        "--selection", str(DEFAULT_SELECTION),
    ]
    sys.argv[1:1] = defaults
    main()
