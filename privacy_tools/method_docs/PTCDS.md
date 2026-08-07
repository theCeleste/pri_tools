# PTCDS

本目录包含 PTCDS 的研究实现，攻击代码位于 `attack/`。本方法和 base 对比微调均调用 `finetune_llama2_sst2_lora_data.py`，分别读取 epsilon 6 的扰动数据和原始 SST-2 数据；直接评估按受保护/未保护两路执行。请从项目根目录运行 `python3 privacy_tool.py`。
