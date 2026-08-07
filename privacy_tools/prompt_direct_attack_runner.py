from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_prompt(path: Path, limit: int, text_field: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("DADP-ICL 示例文件必须是 JSON 列表")
    lines = ["Example:"]
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        text = item.get(text_field) or item.get("text")
        label = item.get("label", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        if isinstance(text, str):
            lines.append(f"Text: {text}\nLabel: {label}")
    lines.append("Continue by writing one plausible private training example.\nText:")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对 DADP-ICL 私有示例上下文执行前缀抽取攻击")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--prompt-examples", type=int, default=5)
    parser.add_argument("--text-field", choices=("synthetic_text", "original_text"), default="synthetic_text")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("--samples 和 --batch-size 必须大于 0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    prompt = build_prompt(args.prompt_data, args.prompt_examples, args.text_field)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    while len(records) < args.samples:
        size = min(args.batch_size, args.samples - len(records))
        encoded = tokenizer([prompt] * size, return_tensors="pt", padding=True).to(device)
        prompt_tokens = encoded.input_ids.shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuations = tokenizer.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True)
        for text in continuations:
            records.append({"sample_id": len(records) + 1, "text": text.strip()})
        print(f"已生成 {len(records)}/{args.samples}")

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "attack": "dadp_icl_prompt_prefix_extraction",
        "model": args.model,
        "prompt_data": str(args.prompt_data.resolve()),
        "text_field": args.text_field,
        "requested_samples": args.samples,
        "generated_samples": len(records),
        "output": str(args.output.resolve()),
    }
    args.output.with_name("attack_generation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
