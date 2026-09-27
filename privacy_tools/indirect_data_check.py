"""Data provenance checks executed by the selected backend's Python environment."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
LORA = PROJECT / "evaluate/indirect_LoRA"


def digest(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_key(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("样本正文为空或不是字符串")
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip()).casefold().encode("utf-8")).hexdigest()


def rows(file):
    file = Path(file)
    if file.suffix == ".tsv":
        with file.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    return json.loads(file.read_text(encoding="utf-8"))


def known_training(plan):
    """Bind full training text hashes to exact adapter bytes and model/dataset."""
    r = plan["resources"]
    adapter = Path(r["adapter"]) / "adapter_model.safetensors"
    checksum = digest(adapter)
    manifest = Path(r["training_manifest"])
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m.get("model") != plan["model"] or m.get("dataset") != plan["dataset"] or m.get("adapter_sha256") != checksum:
            raise ValueError("training_manifest 与模型/数据集/adapter 权重指纹不匹配")
        keys = m.get("training_text_sha256")
        if not m.get("source") or not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", k) for k in keys):
            raise ValueError("training_manifest 必须包含 source 和完整训练文本规范化 SHA256 列表")
        return set(keys), {"training_manifest": str(manifest), "manifest_sha256": digest(manifest), "adapter_sha256": checksum}
    # A bundled adapter (including an identical external copy) can use the
    # delivery's documented and checksummed full training split as evidence.
    checks = json.loads((LORA / "checksums.json").read_text())["files"]
    key = f"weights/{plan['model']}/{plan['dataset']}/adapter_model.safetensors"
    if checks.get(key) != checksum:
        raise ValueError(f"无法核实成员来源：请提供 {manifest}，或配置 evaluation.indirect.models.{plan['model']}.training_manifests.{plan['dataset']}")
    train = LORA / "data" / plan["dataset"] / "train"
    entries = {name: value for name, value in checks.items() if name.startswith(f"data/{plan['dataset']}/train/")}
    if not entries or any(not (LORA / name).is_file() or digest(LORA / name) != value for name, value in entries.items()):
        raise ValueError("随包训练快照缺失或校验不匹配，不能用作成员来源证据")
    from datasets import load_from_disk
    column = "sentence" if plan["dataset"] == "sst2" else "text"
    keys = {text_key(t) for t in load_from_disk(str(train))[column]}
    return keys, {"training_snapshot": str(train), "snapshot_hashes": entries, "adapter_sha256": checksum}


def validate_data(plan):
    r = plan["resources"]
    files = {}
    if plan["finetune_type"] == "explicit":
        column = "sentence" if plan["dataset"] == "sst2" else "text"
        if "member_file" in r:
            member_rows, nonmember_rows = rows(r["member_file"]), rows(r["nonmember_file"])
            files = {r[k]: digest(r[k]) for k in ("member_file", "nonmember_file")}
        else:
            from datasets import load_from_disk
            root = Path(r["dataset_dir"])
            member_rows = load_from_disk(str(root / "train"))
            nonmember_rows = load_from_disk(str(root / "nonmember"))
            files = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
        members = [text_key(row[column]) for row in member_rows]
        others = [text_key(row[column]) for row in nonmember_rows]
        if min(len(members), len(others)) < 1200:
            raise ValueError("LoRA 数据不足，每组必须至少 1200 条；不自动缩减正式样本数")
        if len(set(members[:1200])) != 1200 or len(set(others[:1200])) != 1200:
            raise ValueError("选取的 1200 条样本存在归一化重复；请核对成员划分，不静默重采样")
        training, proof = known_training(plan)
        if not set(members[:1200]) <= training:
            raise ValueError("统一数据前 1200 条候选成员并非全部属于该 adapter 训练集；请校正数据划分")
        if set(others[:1200]) & (training | set(members)):
            raise ValueError("非成员样本与完整训练集或成员数据存在交集")
        return {"member_count": 1200, "nonmember_count": 1200, "data_sha256": files,
                "selected_member_sha256": members[:1200], "selected_nonmember_sha256": others[:1200],
                "membership_definition": "weight_training", **proof}
    root = Path(r["dataset_dir"])
    if plan["dataset"] == "sst2":
        splits = {s: rows(root / name) for s, name in (("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv"))}
        for name in ("train.tsv", "dev.tsv", "test.tsv"):
            files[str(root / name)] = digest(root / name)
        for split in ("train", "validation"):
            if len(splits[split]) < 20 or any(str(row.get("label")) not in {"0", "1"} for row in splits[split]):
                raise ValueError(f"ICL {split} 需至少20条且包含0/1情感标签")
        train_keys = {text_key(row["sentence"]) for row in splits["train"]}
        others = [text_key(row["sentence"]) for row in splits["test"] if isinstance(row.get("sentence"), str) and len(row["sentence"]) > 10][:500]
    else:
        # Use the same preparation function as the native pipeline instead of
        # reproducing WikiText filtering and cross-split deduplication here.
        import sys
        sys.path.insert(0, str(PROJECT / "evaluate/indirect_ICL/DPICL"))
        from task_data import load_wikitext2
        data, _ = load_wikitext2(str(root), "wikitext-2-raw-v1")
        train_keys = {text_key(row["sentence"]) for row in data["train"]}
        others = [text_key(row["sentence"]) for row in data["test"]][:500]
        files = {str(p): digest(p) for p in root.glob("*.json")}
    if len(others) != 500 or len(set(others)) != 500:
        raise ValueError("ICL 需要500条不同的有效非成员；不自动缩减正式样本数")
    if train_keys & set(others):
        raise ValueError("ICL 非成员与示例采样池重叠")
    if r.get("input_json"):
        examples = rows(r["input_json"])
        if not isinstance(examples, list) or len(examples) != 10:
            raise ValueError("复用示例 JSON 必须包含10条记录")
        for row in examples:
            if text_key(row["original_text"]) not in train_keys:
                raise ValueError("示例 original_text 不属于所配置的数据集采样池")
            text_key(row["synthetic_text"])
        files[r["input_json"]] = digest(r["input_json"])
    return {"nonmember_count": 500, "requested_demonstrations": 10,
            "membership_definition": "original_text_records_context_protocol",
            "data_sha256": files, "selected_nonmember_sha256": others}


def check_runtime(kind):
    root = PROJECT / "evaluate" / ("indirect_LoRA" if kind == "explicit" else "indirect_ICL")
    requirements = (root / "requirements.txt").read_text().splitlines()
    if kind == "explicit":
        requirements.append("torch==2.5.1")
    mismatches = []
    for line in requirements:
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.strip().split("==")
        try:
            actual = importlib.metadata.version(name).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            mismatches.append(f"{name}: {actual} (需要 {expected})")
    if mismatches:
        raise ValueError("请为对应后端配置独立 Python 环境：" + "; ".join(mismatches))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("所选物理 GPU 不可用；正式攻击不回退到 CPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.manifest).read_text())
    receipt = validate_data(plan)
    check_runtime(plan["finetune_type"])
    target = Path(plan["output"]) / "data_validation.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Data provenance and runtime checks passed: {target}")


if __name__ == "__main__":
    main()
