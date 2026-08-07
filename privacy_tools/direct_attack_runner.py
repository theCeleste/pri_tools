from __future__ import annotations

import argparse
import json
import math
import shutil
import zlib
from pathlib import Path


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对一个明确指定的受保护 CausalLM adapter 执行前缀抽取攻击"
    )
    parser.add_argument("--model", required=True, help="基础 CausalLM 路径或 Hugging Face 标识")
    parser.add_argument("--adapter", required=True, help="本方法训练得到的 PEFT adapter 目录")
    parser.add_argument("--samples", type=int, required=True, help="精确生成的样本数")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=255)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples 必须大于 0")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    adapter = Path(args.adapter).expanduser().resolve()
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        parser.error(f"不是有效的 PEFT adapter 目录：{adapter}")

    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    runtime_adapter = adapter
    projected_from_sequence_classifier = False
    if adapter_config.get("task_type") != "CAUSAL_LM" or adapter_config.get("modules_to_save"):
        # 这些方法训练的是序列分类 adapter。抽取攻击只应把共享 Transformer
        # 主干上的 LoRA 增量施加到生成模型，不能把分类 head 冒充语言模型 head。
        runtime_adapter = args.output.parent / "attack_adapter_runtime"
        runtime_adapter.mkdir(parents=True, exist_ok=True)
        runtime_config = dict(adapter_config)
        runtime_config["task_type"] = "CAUSAL_LM"
        runtime_config["modules_to_save"] = None
        (runtime_adapter / "adapter_config.json").write_text(
            json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        weight_names = ("adapter_model.safetensors", "adapter_model.bin")
        copied = False
        for name in weight_names:
            source_weight = adapter / name
            if source_weight.is_file():
                shutil.copy2(source_weight, runtime_adapter / name)
                copied = True
        if not copied:
            parser.error(f"adapter 缺少权重文件：{adapter}")
        projected_from_sequence_classifier = True

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"加载受保护模型：base={args.model} adapter={adapter} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(runtime_adapter)).to(device)
    model.eval()

    def perplexity(text: str) -> float:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded.input_ids.to(device)
        if input_ids.shape[1] < 2:
            return float("inf")
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=input_ids).loss
        return float(torch.exp(loss).detach().cpu())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    generated = 0
    while generated < args.samples:
        current_batch = min(args.batch_size, args.samples - generated)
        prompts = [tokenizer.eos_token or ""] * current_batch
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            sequences = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        for text in texts:
            ppl = perplexity(text)
            lower_ppl = perplexity(text.lower())
            zlib_bytes = max(1, len(zlib.compress(text.encode("utf-8"))))
            records.append({
                "sample_id": len(records) + 1,
                "text": text,
                "perplexity": _finite(ppl),
                "lowercase_perplexity": _finite(lower_ppl),
                "ppl_lower_ratio": _finite(ppl / lower_ppl) if lower_ppl else None,
                "ppl_zlib_ratio": _finite(ppl / zlib_bytes),
                "zlib_bytes": zlib_bytes,
            })
        generated += current_batch
        print(f"已生成 {generated}/{args.samples}")

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "attack": "prefix_extraction",
        "model": args.model,
        "adapter": str(adapter),
        "runtime_adapter": str(runtime_adapter),
        "adapter_projection": (
            "backbone_lora_only; classifier_head_excluded"
            if projected_from_sequence_classifier
            else "native_causal_lm_adapter"
        ),
        "requested_samples": args.samples,
        "generated_samples": len(records),
        "seed": args.seed,
        "methodology": "single_protected_model; no duplicate-model ratio",
        "samples_file": str(args.output),
    }
    summary_path = args.output.with_name("attack_generation.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
