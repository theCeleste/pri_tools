from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
from typing import Any, ContextManager


def load_tsv(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [row["sentence"].strip() for row in rows if row.get("sentence", "").strip()]


def load_json_texts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"JSON 数据必须是列表：{path}")
    texts: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = next(
            (item.get(key) for key in ("original_text", "text", "sentence") if isinstance(item.get(key), str)),
            None,
        )
        if text is None and isinstance(item.get("question"), str):
            match = re.search(r'Sentence:\s*["“](.*?)["”]', item["question"], flags=re.DOTALL)
            text = match.group(1) if match else item["question"]
        if text and text.strip():
            texts.append(text.strip())
    return texts


def build_dadp_prompt(path: Path, limit: int) -> str:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    lines = ["Example:"]
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        text = item.get("synthetic_text") or item.get("text")
        label = item.get("label", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        if isinstance(text, str):
            lines.append(f"Text: {text}\nLabel: {label}")
    lines.append("Please analyze this text:\n")
    return "\n".join(lines)


def auc_probability(member: list[float], nonmember: list[float]) -> float:
    wins = 0.0
    total = len(member) * len(nonmember)
    for positive in member:
        for negative in nonmember:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / total if total else 0.0


def quantile_higher(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def metric_rows(member: list[float], nonmember: list[float], targets: list[float]) -> list[dict[str, float]]:
    auc = auc_probability(member, nonmember)
    rows: list[dict[str, float]] = []
    for target in targets:
        threshold = quantile_higher(nonmember, 1.0 - target)
        rows.append({
            "target_fpr": target,
            "threshold": threshold,
            "tpr": sum(score > threshold for score in member) / len(member),
            "fpr": sum(score > threshold for score in nonmember) / len(nonmember),
            "auc": auc,
        })
    return rows


def self_test(output_dir: Path) -> int:
    member = [0.9, 0.8, 0.7, 0.6]
    nonmember = [0.4, 0.3, 0.2, 0.1]
    rows = metric_rows(member, nonmember, [0.25, 0.1])
    if rows[0]["auc"] != 1.0:
        raise AssertionError("AUC 自检失败")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "self_test.json").write_text(
        json.dumps({"passed": True, "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "rows": rows}, indent=2))
    return 0


def projected_adapter(adapter: Path, output_dir: Path) -> tuple[Path, str]:
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"不是有效的 PEFT adapter：{adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("task_type") == "CAUSAL_LM" and not config.get("modules_to_save"):
        return adapter, "native_causal_lm_adapter"
    runtime = output_dir / "attack_adapter_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    updated = dict(config)
    updated["task_type"] = "CAUSAL_LM"
    updated["modules_to_save"] = None
    (runtime / "adapter_config.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (adapter / name).is_file():
            shutil.copy2(adapter / name, runtime / name)
            return runtime, "backbone_lora_only; classifier_head_excluded"
    raise ValueError(f"adapter 缺少权重文件：{adapter}")


def load_steering(wrapper_path: Path, checkpoint: Path, torch: Any) -> Any:
    spec = importlib.util.spec_from_file_location("privacy_tool_steering_wrapper", wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 steering wrapper：{wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    activations = checkpoint_data.get("layer_activations", checkpoint_data)
    return module.FrozenSteering(activations)


def main() -> int:
    parser = argparse.ArgumentParser(description="重新生成邻居分数并执行成员推断攻击")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prompt-data", type=Path)
    parser.add_argument("--prompt-examples", type=int, default=5)
    parser.add_argument("--steering-checkpoint", type=Path)
    parser.add_argument("--steering-wrapper", type=Path)
    parser.add_argument("--steering-multiplier", type=float, default=2.0)
    parser.add_argument("--member-tsv", type=Path)
    parser.add_argument("--member-json", type=Path)
    parser.add_argument("--nonmember-tsv", type=Path)
    parser.add_argument("--mask-model", default="roberta-base")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        return self_test(args.output_dir)
    if not args.method or not args.model or not args.nonmember_tsv:
        parser.error("实际攻击需要 --method、--model 和 --nonmember-tsv")
    if bool(args.member_tsv) == bool(args.member_json):
        parser.error("必须且只能指定 --member-tsv 或 --member-json")
    protection_modes = sum(bool(value) for value in (args.adapter, args.prompt_data, args.steering_checkpoint))
    if protection_modes != 1:
        parser.error("必须且只能指定 adapter、DADP-ICL prompt 或 PrivSV steering 中的一种")
    if args.steering_checkpoint and not args.steering_wrapper:
        parser.error("PrivSV 攻击还需要 --steering-wrapper")
    if args.samples < 1 or args.neighbors < 1:
        parser.error("--samples 和 --neighbors 必须大于 0")

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    victim_tokenizer = AutoTokenizer.from_pretrained(args.model)
    victim_tokenizer.truncation_side = "left"
    if victim_tokenizer.pad_token_id is None:
        victim_tokenizer.pad_token = victim_tokenizer.eos_token
    victim = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    adapter_mode = None
    if args.adapter:
        from peft import PeftModel

        runtime_adapter, adapter_mode = projected_adapter(args.adapter.resolve(), args.output_dir)
        victim = PeftModel.from_pretrained(victim, str(runtime_adapter))
    victim = victim.to(device).eval()

    prompt = build_dadp_prompt(args.prompt_data, args.prompt_examples) if args.prompt_data else ""
    steering = (
        load_steering(args.steering_wrapper.resolve(), args.steering_checkpoint.resolve(), torch)
        if args.steering_checkpoint
        else None
    )
    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForMaskedLM.from_pretrained(args.mask_model).to(device).eval()
    if mask_tokenizer.mask_token is None:
        raise ValueError(f"掩码模型没有 mask token：{args.mask_model}")

    member_pool = load_tsv(args.member_tsv) if args.member_tsv else load_json_texts(args.member_json)
    nonmember_pool = load_tsv(args.nonmember_tsv)
    count = min(args.samples, len(member_pool), len(nonmember_pool))
    if count < 1:
        raise ValueError("成员或非成员数据为空")
    members = random.sample(member_pool, count)
    nonmembers = random.sample(nonmember_pool, count)

    def steering_context(prompt_len: int) -> ContextManager[None]:
        if steering is None:
            return contextlib.nullcontext()
        return steering.apply(victim, multiplier=args.steering_multiplier, min_token_index=prompt_len)

    def log_probability(text: str) -> float:
        full_text = prompt + text
        encoded = victim_tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=args.max_length
        ).to(device)
        input_ids = encoded.input_ids
        target_len = victim_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.max_length
        ).input_ids.shape[1]
        target_len = min(max(1, target_len), max(1, input_ids.shape[1] - 1))
        prompt_len = max(0, input_ids.shape[1] - target_len)
        with torch.no_grad(), steering_context(prompt_len):
            logits = victim(input_ids=input_ids).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, reduction="none"
        )
        relevant = losses[:, -target_len:]
        return float((-relevant.mean()).detach().cpu())

    def neighbours(text: str) -> list[str]:
        words = text.split()
        if len(words) < 2:
            return [text]
        positions = list(range(len(words)))
        random.shuffle(positions)
        generated: list[str] = []
        for position in positions:
            masked = list(words)
            masked[position] = mask_tokenizer.mask_token
            encoded = mask_tokenizer(
                " ".join(masked), return_tensors="pt", truncation=True, max_length=args.max_length
            ).to(device)
            mask_locations = (encoded.input_ids == mask_tokenizer.mask_token_id).nonzero(as_tuple=False)
            if len(mask_locations) == 0:
                continue
            with torch.no_grad():
                logits = mask_model(**encoded).logits
            token_index = int(mask_locations[0, 1])
            candidates = torch.topk(logits[0, token_index], k=12).indices.tolist()
            for candidate in candidates:
                if candidate in mask_tokenizer.all_special_ids:
                    continue
                replacement = mask_tokenizer.decode([candidate]).strip()
                if replacement and replacement.casefold() != words[position].casefold() and replacement.isprintable():
                    changed = list(words)
                    changed[position] = replacement
                    generated.append(" ".join(changed))
                    break
            if len(generated) >= args.neighbors:
                break
        return generated or [text]

    records: list[dict[str, object]] = []
    for membership, texts in ((1, members), (0, nonmembers)):
        for index, text in enumerate(texts):
            variants = neighbours(text)
            original_score = log_probability(text)
            neighbor_scores = [log_probability(item) for item in variants]
            score = original_score - sum(neighbor_scores) / len(neighbor_scores)
            records.append({
                "membership": membership,
                "sample_index": index,
                "text": text,
                "original_log_probability": original_score,
                "mean_neighbor_log_probability": sum(neighbor_scores) / len(neighbor_scores),
                "membership_score": score,
                "neighbors_generated": len(variants),
            })
            print(f"{args.method}: 已处理 {len(records)}/{count * 2}")

    member_scores = [float(row["membership_score"]) for row in records if row["membership"] == 1]
    nonmember_scores = [float(row["membership_score"]) for row in records if row["membership"] == 0]
    metrics = metric_rows(member_scores, nonmember_scores, args.target_fprs)
    csv_path = args.output_dir / "fresh_indirect_attack_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema_version": 1,
        "attack": "fresh_neighbour_membership_inference",
        "fresh_scores_generated": True,
        "method": args.method,
        "protection_mode": "adapter" if args.adapter else "dadp_icl_prompt" if args.prompt_data else "privsv_steering",
        "adapter_projection": adapter_mode,
        "model": args.model,
        "mask_model": args.mask_model,
        "member_samples": count,
        "nonmember_samples": count,
        "neighbors_per_sample": args.neighbors,
        "metrics": metrics,
        "scores_csv": str(csv_path.resolve()),
    }
    (args.output_dir / "fresh_indirect_attack.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
