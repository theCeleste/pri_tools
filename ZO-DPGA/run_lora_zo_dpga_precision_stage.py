import sys

from run_lora_zo_dpga_precision import DEFAULT_CONFIG, DEFAULT_SELECTION
from run_lora_zo_dpga_stage import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--runner-module", "run_lora_zo_dpga_precision",
        "--config", str(DEFAULT_CONFIG),
        "--selection", str(DEFAULT_SELECTION),
    ]
    main()
