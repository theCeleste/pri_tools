import sys

from run_lora_zo_dpga_stage import main
from run_lora_mezo_refinement import DEFAULT_CONFIG, DEFAULT_SELECTION


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--runner-module", "run_lora_mezo_refinement",
        "--config", str(DEFAULT_CONFIG), "--selection", str(DEFAULT_SELECTION),
    ]
    main()
