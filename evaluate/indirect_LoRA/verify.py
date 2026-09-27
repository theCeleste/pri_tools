"""Validate the delivery files, dependencies, datasets and external base models."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from datasets import load_from_disk
from safetensors import safe_open

from run import DATASETS, MODELS, ROOT, model_path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "checksums.json").read_text())
    present = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
               if path.is_file() and "runs" not in path.relative_to(ROOT).parts
               and "__pycache__" not in path.parts and path.name != "checksums.json"}
    if present != set(manifest["files"]):
        raise RuntimeError(f"package file list mismatch: missing={sorted(set(manifest['files']) - present)}, "
                           f"extra={sorted(present - set(manifest['files']))}")
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"file missing or changed: {name}")
    print(f"File checksums: {len(manifest['files'])} OK", flush=True)

    for name, expected in [("torch", "2.5.1"), *(line.split("==") for line in
                            (ROOT / "requirements.txt").read_text().splitlines() if line.strip())]:
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"dependency mismatch: {name} {actual} != {expected}")
    print("Dependency versions: OK", flush=True)

    for dataset, (column, _) in DATASETS.items():
        expected = {"sst2": (33579, 33399), "wikitext2": (8470, 8471)}[dataset]
        lengths = tuple(len(load_from_disk(str(ROOT / "data" / dataset / split)))
                        for split in ("train", "nonmember"))
        if lengths != expected:
            raise RuntimeError(f"{dataset} split sizes: {lengths} != {expected}")
        for split in ("train", "nonmember"):
            data = load_from_disk(str(ROOT / "data" / dataset / split))
            if column not in data.column_names or any(not str(text).strip()
                                                       for text in data.select(range(1200))[column]):
                raise RuntimeError(f"{dataset}/{split}: missing or empty text")
        print(f"{dataset} data: {lengths} OK", flush=True)

    for model, (folder, _) in MODELS.items():
        base = model_path(model)
        if not (base / "config.json").is_file():
            raise RuntimeError(f"base model missing: {base}")
        if model == "fm9g":
            if not (base / "pytorch_model.bin").is_file():
                raise RuntimeError(f"base model weights missing: {base}")
        else:
            index = json.loads((base / "model.safetensors.index.json").read_text())
            if any(not (base / name).is_file() for name in set(index["weight_map"].values())):
                raise RuntimeError(f"base model shard missing: {base}")
        for dataset in DATASETS:
            adapter = ROOT / "weights" / model / dataset
            config = json.loads((adapter / "adapter_config.json").read_text())
            if config["base_model_name_or_path"] != f"../models/{folder}":
                raise RuntimeError(f"wrong adapter base: {adapter}")
            with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
                if not list(weights.keys()):
                    raise RuntimeError(f"empty adapter: {adapter}")
        print(f"{model} base and adapters: OK", flush=True)


if __name__ == "__main__":
    main()
