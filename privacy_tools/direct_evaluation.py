"""Unified direct evaluation for the ChatGLM3 and FM9G backends.

The menu process deliberately imports only the standard library and ``core``.
Model code is always run in a child process, which keeps preview/history useful
on machines where torch or transformers are not installed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from uuid import uuid4

from .core import PROJECT_ROOT, configured_python, get_nested, load_settings, project_path
from .evaluation_history import archive_result, load_result

ENGINE = "evaluate-direct-v1"
ROOT = PROJECT_ROOT / "evaluate/direct"
GENERATOR = {
    "chatglm3": ROOT / "lora/chatglm/temperature_chatglm.py",
    "fm9g": ROOT / "lora/9g/temperature_9g.py",
}
MODEL_INFO = {
    "chatglm3": ("CHATGLM_MODEL_DIR", "chatglm3_6b", "common.chatglm3_6b"),
    "fm9g": ("FM9G_MODEL_DIR", "9G7B_MHA", "common.fm9g_7b"),
}
DATASET_ALIASES = {"wikitext": "wikitext2", "wiki": "wikitext2", "wiki2": "wikitext2"}
SCRIPT_DATASETS = {"sst2": "sst2", "wikitext2": "wikitext"}


def path(value: str | Path) -> Path:
    return project_path(str(value)).resolve()


def read_json(file: str | Path):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write_json(file: str | Path, value) -> None:
    target = Path(file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(model: str, kind: str, dataset: str) -> dict[str, str]:
    model = {"chatglm": "chatglm3", "fm9g-7b": "fm9g"}.get(model, model)
    dataset = DATASET_ALIASES.get(dataset, dataset)
    if model not in MODEL_INFO or kind not in {"explicit", "implicit"} or dataset not in {"sst2", "wikitext2"}:
        raise ValueError(f"不支持的直接评估组合：{model}/{kind}/{dataset}")
    return {"model": model, "finetune_type": kind, "dataset": dataset, "engine": ENGINE}


def result_root(settings: dict, ident: dict) -> Path:
    return path(settings.get("results_dir", "privacy_tools/results")) / "direct_evaluation" / ident["model"] / ident["finetune_type"] / ident["dataset"]


def backend_digest(kind: str) -> str:
    files = [Path(__file__)]
    if kind == "explicit":
        files.extend(GENERATOR.values())
    else:
        files.extend((ROOT / "DPICL/run_exp_sst2_local.py", ROOT / "DPICL/member_test_current.py"))
    h = hashlib.sha256()
    for file in files:
        if file.is_file():
            h.update(str(file.relative_to(PROJECT_ROOT) if file.is_relative_to(PROJECT_ROOT) else file).encode())
            h.update(file.read_bytes())
    return h.hexdigest()


def _option(settings: dict, key: str, default=None):
    return get_nested(settings, f"evaluation.direct.{key}", default)


def _model_option(settings: dict, model: str, key: str, default=None):
    return _option(settings, f"models.{model}.{key}", default)


def _dataset_option(settings: dict, key: str, dataset: str, default=None):
    value = _option(settings, f"datasets.{dataset}.{key}", None)
    return default if value is None else value


def build_plan(settings: dict, model: str, kind: str, dataset: str, output: str | Path, *, source_run: str | Path | None = None) -> dict:
    """Build a serializable execution plan without touching model libraries."""
    ident = identity(model, kind, dataset)
    model, dataset = ident["model"], ident["dataset"]
    output = path(output)
    backend = output / "backend"
    # Keep the direct backends isolated like the indirect evaluation runner:
    # LoRA generation/matching and ICL generation/matching may require
    # different environments and dependency versions.
    python_profile = "direct_lora" if kind == "explicit" else "direct_icl"
    py = configured_python(settings, python_profile)
    env_key, model_name, common_key = MODEL_INFO[model]
    configured = _model_option(settings, model, "base_model") or get_nested(settings, common_key)
    base_value = os.environ.get(env_key) or configured or str(ROOT / "models" / model_name)
    base = path(base_value)
    # GPU selection is shared by all evaluation backends.  Direct evaluation
    # intentionally does not have an evaluation.direct.gpu override.
    gpu = str(get_nested(settings, "common.gpu", "0")).strip()
    if "," in gpu:
        gpu = gpu.split(",", 1)[0].strip()
    cpu_requested = gpu.lower() in {"cpu", "-1", "none", ""}
    # CUDA_VISIBLE_DEVICES remaps the selected physical card to cuda:0 inside
    # the child process. Keep the configured value as provenance, but never
    # pass the physical index through as a logical torch device index.
    visible_gpu = "" if cpu_requested else gpu
    logical_device = "cpu" if cpu_requested else "cuda:0"
    seed = int(_option(settings, "seed", 0))
    data_default = get_nested(settings, "common.dataset_path", "privacy_tools/datasets/sst2")
    data_value = _dataset_option(settings, "path", dataset, data_default)
    data = path(data_value)
    adapter = path(_model_option(settings, model, f"adapters.{dataset}", ROOT / "weights" / model / dataset))
    tokenizer = path(_model_option(settings, model, f"tokenizers.{dataset}", adapter))
    candidates = backend / ("candidates_top100.txt" if kind == "explicit" else "generated.json")
    generated_json = backend / "candidates_all.json" if kind == "explicit" else backend / "generated.json"
    match_dir = backend / "matching"
    resources = {
        "base_model": str(base), "adapter": str(adapter), "tokenizer": str(tokenizer),
        "dataset": str(data), "candidate_file": str(candidates), "generated_json": str(generated_json),
        "match_dir": str(match_dir), "physical_gpu": gpu, "logical_device": logical_device, "python": py,
    }
    warnings: list[str] = []
    if str(base_value).startswith(("THUDM/", "OpenBMB/")):
        warnings.append(f"{env_key} 或 evaluation.direct.models.{model}.base_model 必须指定本地模型目录，当前值为 {base_value}")
    env = {
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": visible_gpu,
        "PYTHONHASHSEED": str(seed),
    }
    stages: list[tuple[str, list[str]]] = []
    if source_run:
        source = path(source_run)
        summary_file = source / "summary.json"
        if not summary_file.is_file():
            raise ValueError(f"离线重算来源缺少 summary.json：{source}")
        summary = read_json(summary_file)
        if any(summary.get(k) != v for k, v in ident.items()) or summary.get("status") != "completed":
            raise ValueError("离线重算来源必须是模型、微调方式和数据集匹配的新版成功运行目录")
        raw = source / "backend"
        candidate = raw / ("candidates_top100.txt" if kind == "explicit" else "generated.json")
        if not candidate.is_file():
            candidate = raw / ("candidates_all.json" if kind == "explicit" else "generated.json")
        if not candidate.is_file():
            raise ValueError(f"离线来源缺少候选结果：{raw}")
        resources["candidate_file"] = str(candidate)
    else:
        if kind == "explicit":
            args = ["--base_model_path", str(base), "--tokenizer_path", str(tokenizer), "--lora_path", str(adapter), "--dataset", SCRIPT_DATASETS[dataset],
                    "--N", str(_option(settings, "generation.candidates", 1000)), "--batch_size", str(_option(settings, "generation.batch_size", 5 if model == "chatglm3" else 10)),
                    "--seq_len", str(_option(settings, "generation.max_new_tokens", 180)), "--top_p", str(_option(settings, "generation.top_p", 0.6)),
                    "--top_k", str(_option(settings, "generation.top_k", 20)), "--dtype", str(_option(settings, "generation.dtype", "auto")),
                    "--min_chars", str(_option(settings, "generation.min_chars", 20)), "--outfile", str(backend / "candidates"), "--seed", str(seed)]
            if model == "fm9g":
                args += ["--decode_mode", str(_option(settings, "generation.decode_mode", "sample"))]
            stages.append(("generation", [py, str(GENERATOR[model]), *args]))
        else:
            num_valid = _option(settings, "generation.num_valid", None)
            if num_valid is None:
                # Keep compatibility with the pre-existing configuration key.
                num_valid = _option(settings, "generation.num_demonstrations", 100)
            args = ["--sigma", "0", "--openai-model", str(base), "--task", SCRIPT_DATASETS[dataset], "--data-dir", str(data),
                    "--device", logical_device, "--output-file", str(generated_json), "--num-valid", str(num_valid),
                    "--num-test", str(_option(settings, "generation.num_queries", 100)), "--max-token-per-text", str(_option(settings, "generation.max_new_tokens", 100)),
                    "--synth-seed", str(seed), "--eval-seed", str(seed), "--num-private-train", "0", "--num-private-train-splits", "0", "--no-noise"]
            stages.append(("generation_no_noise", [py, str(ROOT / "DPICL/run_exp_sst2_local.py"), *args]))
    if kind == "implicit":
        # Keep implicit matching on the original DPICL member-test path.  The
        # script writes the canonical matches/summary files consumed below.
        candidate_field = "generated_text" if dataset == "wikitext2" else "synthetic_text"
        match_cmd = [py, str(ROOT / "DPICL/member_test_current.py"),
                     "--task", SCRIPT_DATASETS[dataset],
                     "--generated-json", resources["candidate_file"],
                     "--train-data", str(data),
                     "--candidate-field", candidate_field,
                     "--output-dir", str(match_dir),
                     "--fuzzy-threshold", str(_option(settings, "matching.fuzzy_threshold", 0.85)),
                     "--contains-min-chars", str(_option(settings, "matching.contains_min_chars", 50)),
                     "--min-chars", str(_option(settings, "matching.min_chars", 1)),
                     "--window-min-tokens", str(_option(settings, "matching.window_min_tokens", 8))]
    else:
        match_cmd = [py, "-m", "privacy_tools.direct_evaluation", "--match", "--dataset", dataset,
                     "--generated", resources["candidate_file"], "--train-data", str(data), "--output-dir", str(match_dir),
                     "--fuzzy-threshold", str(_option(settings, "matching.fuzzy_threshold", 0.85)),
                     "--contains-min-chars", str(_option(settings, "matching.contains_min_chars", 50)),
                     "--min-chars", str(_option(settings, "matching.min_chars", 1)),
                     "--window-min-tokens", str(_option(settings, "matching.window_min_tokens", 8))]
    stages.append(("matching", match_cmd))
    return {**ident, "backend_digest": backend_digest(kind), "output": str(output), "backend": str(backend), "python": py,
            "env": env, "resources": resources, "warnings": warnings, "stages": stages,
            "source_run": str(path(source_run)) if source_run else None,
            "no_noise": kind == "implicit", "seed": seed}


def _weight_files(folder: Path) -> list[Path]:
    index = next((folder / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if (folder / name).is_file()), None)
    if index:
        try:
            return [folder / name for name in set(read_json(index)["weight_map"].values())]
        except (KeyError, TypeError, ValueError):
            return []
    return [folder / name for name in ("model.safetensors", "pytorch_model.bin") if (folder / name).is_file()]


def require_model(folder: str | Path, *, adapter: bool = False, tokenizer: bool = False) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"模型目录不存在：{folder}")
    if tokenizer:
        if not (folder / "tokenizer_config.json").is_file() or not any((folder / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.txt", "vocab.json")):
            raise ValueError(f"缺少 tokenizer 文件：{folder}")
        return
    config = folder / ("adapter_config.json" if adapter else "config.json")
    if not config.is_file():
        raise ValueError(f"缺少模型配置：{config}")
    if adapter:
        names = [next((folder / name for name in ("adapter_model.safetensors", "adapter_model.bin") if (folder / name).is_file()), folder / "adapter_model.safetensors")]
    else:
        names = _weight_files(folder)
    if not names or any(not file.is_file() or file.stat().st_size == 0 for file in names):
        raise ValueError(f"缺少或为空的{'LoRA' if adapter else '模型'}权重：{folder}")


def require_dataset(dataset: str, value: str | Path) -> None:
    folder = Path(value)
    if not folder.exists():
        raise ValueError(f"数据集路径不存在：{folder}")
    if dataset == "sst2":
        if folder.is_file():
            return
        if not any((folder / name).is_file() for name in ("train.tsv", "train.csv", "train.json", "train.jsonl", "dataset_dict.json")):
            raise ValueError(f"SST-2 数据集缺少 train 文件：{folder}")
    elif not (
        folder.is_file()
        or (folder / "dataset_dict.json").is_file()
        or any((folder / name).is_file() for name in ("train.json", "train.jsonl"))
        or any(folder.rglob("*.txt"))
        or any(folder.rglob("*.parquet"))
    ):
        raise ValueError(f"WikiText-2 数据集缺少 train.json、文本或 parquet 文件：{folder}")


def preflight(plan: dict) -> None:
    if not shutil.which(plan["python"]):
        profile = "python.direct_lora" if plan["finetune_type"] == "explicit" else "python.direct_icl"
        raise ValueError(f"Python 不可用：{plan['python']}；请配置 {profile}")
    if plan["source_run"]:
        return
    if plan["warnings"]:
        raise ValueError("; ".join(plan["warnings"]))
    resources = plan["resources"]
    require_model(resources["base_model"])
    require_model(resources["base_model"], tokenizer=True)
    require_dataset(plan["dataset"], resources["dataset"])
    if plan["finetune_type"] == "explicit":
        require_model(resources["adapter"], adapter=True)
        require_model(resources["tokenizer"], tokenizer=True)


def normalize_basic(text) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text.replace("#39;", "'"))
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_loose(text) -> str:
    return re.sub(r"[^\w\s]", " ", normalize_basic(text), flags=re.UNICODE).replace("_", " ").strip()


def best_similarity(candidate: str, reference: str) -> float:
    cand, ref = normalize_loose(candidate), normalize_loose(reference)
    if not cand or not ref:
        return 0.0
    scores = [difflib.SequenceMatcher(None, cand, ref).ratio()]
    if len(cand) >= len(ref):
        scores.append(difflib.SequenceMatcher(None, cand[:len(ref)], ref).ratio())
    if len(ref) >= len(cand):
        scores.append(difflib.SequenceMatcher(None, cand, ref[:len(cand)]).ratio())
    return max(scores)


def best_window_similarity(candidate: str, reference: str) -> float:
    """Compare a WikiText continuation against local token windows."""
    candidate_tokens = normalize_loose(candidate).split()
    reference_tokens = normalize_loose(reference).split()
    if not candidate_tokens or not reference_tokens:
        return 0.0
    if len(candidate_tokens) > len(reference_tokens):
        return difflib.SequenceMatcher(None, candidate_tokens, reference_tokens, autojunk=False).ratio()
    matcher = difflib.SequenceMatcher(None, candidate_tokens, reference_tokens, autojunk=False)
    window_size = len(candidate_tokens)
    best = 0.0
    for block in matcher.get_matching_blocks():
        if not block.size:
            continue
        start = max(0, min(block.b, len(reference_tokens) - window_size))
        score = difflib.SequenceMatcher(None, candidate_tokens, reference_tokens[start:start + window_size], autojunk=False).ratio()
        best = max(best, score)
        if best >= 1.0:
            break
    return best


def _read_rows(file: Path) -> list[dict]:
    if file.suffix.lower() in {".json", ".jsonl"}:
        if file.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = read_json(file)
        if isinstance(value, dict):
            for key in ("train", "data", "rows"):
                if isinstance(value.get(key), list):
                    return value[key]
            return [value]
        return value if isinstance(value, list) else []
    if file.suffix.lower() in {".tsv", ".csv"}:
        with file.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if file.suffix.lower() == ".tsv" else ","))
    return [{"text": line.strip()} for line in file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("= ")]


def load_training_records(dataset: str, source: str | Path) -> list[dict]:
    root = Path(source)
    # Some local WikiText/SST-2 exports are HuggingFace DatasetDicts or
    # parquet shards.  Import datasets only in the matching child process.
    if root.is_dir() and (root / "dataset_dict.json").is_file():
        try:
            from datasets import load_from_disk
            loaded = load_from_disk(str(root))
            rows = []
            for item in loaded["train"]:
                rows.append(item)
            source = root / "__dataset_dict_train__"
            return _records_from_rows(dataset, rows)
        except (ImportError, KeyError, OSError, ValueError):
            pass
    if root.is_dir() and not any((root / name).is_file() for name in ("train.tsv", "train.csv", "train.json", "train.jsonl")):
        parquet_files = sorted(root.rglob("*.parquet"))
        if parquet_files:
            train_files = [
                file for file in parquet_files
                if file.name.startswith("train") or "train" in file.relative_to(root).parts[:-1]
            ]
            try:
                from datasets import load_dataset
                loaded = load_dataset("parquet", data_files={"train": [str(file) for file in (train_files or parquet_files)]})
                return _records_from_rows(dataset, list(loaded["train"]))
            except (ImportError, OSError, ValueError):
                pass
    files = [root] if root.is_file() else []
    if not files:
        if dataset == "sst2":
            files = [root / name for name in ("train.tsv", "train.csv", "train.json", "train.jsonl") if (root / name).is_file()]
        else:
            files = [root / name for name in ("train.json", "train.jsonl") if (root / name).is_file()]
            if not files:
                text_files = sorted(root.rglob("*.txt"))
                train_text_files = [file for file in text_files if "train" in file.name.lower()]
                files = train_text_files or text_files
    rows: list[dict] = []
    for file in files:
        for row in _read_rows(file):
            text = row.get("sentence", row.get("text", row.get("content", ""))) if isinstance(row, dict) else str(row)
            if str(text).strip():
                item = {"idx": len(rows), "text": str(text).strip()}
                if isinstance(row, dict) and row.get("label") not in (None, ""):
                    try:
                        item["label"] = int(row["label"])
                    except (TypeError, ValueError):
                        item["label"] = row["label"]
                rows.append(item)
    return _records_from_rows(dataset, rows)


def _records_from_rows(dataset: str, rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        text = row.get("sentence", row.get("text", row.get("content", ""))) if isinstance(row, dict) else str(row)
        if str(text).strip():
            item = {"idx": len(records), "text": str(text).strip()}
            if isinstance(row, dict) and row.get("label") not in (None, ""):
                try:
                    item["label"] = int(row["label"])
                except (TypeError, ValueError):
                    item["label"] = row["label"]
            records.append(item)
    return records


def load_candidates(source: str | Path, dataset: str) -> list[dict]:
    file = Path(source)
    if not file.is_file():
        raise ValueError(f"候选结果不存在：{file}")
    if file.suffix.lower() in {".json", ".jsonl"}:
        if file.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            values = read_json(file)
        if isinstance(values, dict):
            values = values.get("results", values.get("candidates", [values]))
        return [_candidate_dict(value, index, dataset) for index, value in enumerate(values or [])]
    text = file.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*[-=]{10,}\s*\n", text)
    candidates = []
    for block in blocks:
        match = re.search(r"(?:^|\n)\s*Text:\s*(.*?)(?=\n\s*[-=]{10,}|\Z)", block, re.I | re.S)
        if match:
            rank = re.search(r"\[\s*(\d+)\s*\]", block)
            ratio = re.search(r"Ratio:\s*([-+]?\d+(?:\.\d+)?)", block, re.I)
            candidates.append({"candidate_index": len(candidates), "text": match.group(1).strip(), "rank": int(rank.group(1)) if rank else len(candidates) + 1,
                               "ratio": float(ratio.group(1)) if ratio else None})
    return candidates


def _candidate_dict(value, index: int, dataset: str) -> dict:
    if not isinstance(value, dict):
        return {"candidate_index": index, "text": "" if value is None else str(value)}
    field = "generated_text" if dataset == "wikitext2" else "synthetic_text"
    text = value.get("text", value.get(field, value.get("generated", value.get("sample", ""))))
    return {"candidate_index": index, "text": "" if text is None else str(text), **{key: value[key] for key in ("prompt", "ppl", "zlib", "ratio", "label", "original_text") if key in value}}


def match_candidates(candidates: list[dict], train_records: list[dict], *, fuzzy_threshold: float = 0.85, contains_min_chars: int = 50, min_chars: int = 1, dataset: str = "sst2", window_min_tokens: int = 8) -> tuple[list[dict], dict]:
    raw = {item["text"]: item for item in train_records}
    basic = {normalize_basic(item["text"]): item for item in train_records}
    results = []
    counts = Counter()
    for candidate in candidates:
        text = str(candidate.get("text", ""))
        result = {**candidate, "matched": False, "match_type": "unmatched", "score": 0.0, "train_index": None, "matched_text": None, "label": candidate.get("label")}
        if len(text.strip()) >= min_chars:
            record = raw.get(text) or basic.get(normalize_basic(text))
            if record:
                result.update(matched=True, match_type="raw_exact" if raw.get(text) else "normalized_exact", score=1.0, train_index=record["idx"], matched_text=record["text"], label=record.get("label", result.get("label")))
            elif len(normalize_loose(text)) >= contains_min_chars:
                loose = normalize_loose(text)
                record = next((item for item in train_records if loose in normalize_loose(item["text"]) or normalize_loose(item["text"]) in loose), None)
                if record:
                    result.update(matched=True, match_type="contains", score=1.0, train_index=record["idx"], matched_text=record["text"], label=record.get("label", result.get("label")))
            if not result["matched"]:
                best, score = None, 0.0
                for record in train_records:
                    current = (best_window_similarity(text, record["text"])
                               if dataset == "wikitext2" and len(normalize_loose(text).split()) >= window_min_tokens
                               else best_similarity(text, record["text"]))
                    if current > score:
                        best, score = record, current
                if best is not None and score >= fuzzy_threshold:
                    result.update(matched=True, match_type="window_fuzzy" if dataset == "wikitext2" else "fuzzy", score=score, train_index=best["idx"], matched_text=best["text"], label=best.get("label", result.get("label")))
        counts[result["match_type"]] += 1
        results.append(result)
    total = len(results)
    summary = {"candidate_count": total, "matched_count": sum(item["matched"] for item in results), "match_rate": (sum(item["matched"] for item in results) / total if total else 0.0),
               "match_type_counts": dict(counts), "fuzzy_threshold": fuzzy_threshold, "contains_min_chars": contains_min_chars, "min_chars": min_chars,
               "window_min_tokens": window_min_tokens, "dataset": dataset, "empty_candidates": sum(not item.get("text", "").strip() for item in candidates)}
    return results, summary


def run_match(args) -> int:
    candidates = load_candidates(args.generated, args.dataset)
    records = load_training_records(args.dataset, args.train_data)
    if not records:
        raise ValueError(f"未加载到训练集文本：{args.train_data}")
    results, summary = match_candidates(candidates, records, fuzzy_threshold=args.fuzzy_threshold, contains_min_chars=args.contains_min_chars, min_chars=args.min_chars, dataset=args.dataset, window_min_tokens=args.window_min_tokens)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "matches.json", results)
    write_json(output / "summary.json", summary)
    with (output / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("candidate_index", "text", "matched", "match_type", "score", "train_index", "matched_text", "label")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in results)
    return 0


def format_summary(summary: dict, *, include_output: bool = True) -> str:
    lines = [f"模型：{summary['model']}，微调方式：{summary['finetune_type']}，数据集：{summary['dataset']}"]
    metrics = summary.get("matching", {})
    lines.append(f"候选数：{metrics.get('candidate_count', 0)}，匹配数：{metrics.get('matched_count', 0)}，匹配率：{metrics.get('match_rate', 0):.4%}")
    if include_output:
        lines.append(f"结果目录：{summary['output']}")
    return "\n".join(lines)


def print_summary(summary: dict) -> None:
    print(format_summary(summary))


def show_history(settings: dict, ident: dict) -> int:
    history = load_result(settings, "direct", ident)
    if history is None:
        print("未找到当前模型、微调方式和数据集的无扰动历史结果。")
        return 0
    if history["kind"] == "text":
        print(history["text"], end="")
    else:
        print_summary(history["summary"])
    return 0


def run_direct(settings: dict, model: str, kind: str, *, mode: str = "preview", dataset: str | None = None, source_run: str | None = None) -> int:
    try:
        ident = identity(model, kind, dataset or get_nested(settings, "common.dataset", "sst2"))
        if mode == "history":
            return show_history(settings, ident)
        if mode not in {"run", "preview", "evaluate"}:
            raise ValueError(f"未知执行模式：{mode}")
        if (mode == "evaluate") != bool(source_run):
            raise ValueError("离线重算必须且仅能提供 --source-run")
        output = result_root(settings, ident) / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8])
        plan = build_plan(settings, ident["model"], kind, ident["dataset"], output, source_run=source_run)
        for warning in plan["warnings"]:
            print(f"提示：{warning}")
        if mode == "preview":
            print("命令预览（不启动计算、不创建目录）")
            print("环境：" + shlex.join(f"{key}={value}" for key, value in plan["env"].items()))
            for name, command in plan["stages"]:
                print(name + ": " + shlex.join(command))
            return 0
        preflight(plan)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {**plan, "status": "running", "stage_results": []}
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "config.json", {key: plan[key] for key in ("model", "finetune_type", "dataset", "engine", "seed", "no_noise", "resources")})
        write_json(output / "commands.json", {name: command for name, command in plan["stages"]})
        write_json(output / "environment.json", plan["env"])
        env = {**os.environ, **plan["env"]}
        try:
            for name, command in plan["stages"]:
                start = time.monotonic()
                print(f"START {name}; 日志：{output / (name + '.log')}", flush=True)
                with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    returncode = process.wait()
                manifest["stage_results"].append({"name": name, "returncode": returncode, "seconds": time.monotonic() - start})
                write_json(output / "run_manifest.json", manifest)
                if returncode:
                    raise RuntimeError(f"{name} 失败（退出码 {returncode}），查看 {output / (name + '.log')}")
            matching = read_json(Path(plan["resources"]["match_dir"]) / "summary.json")
            completed_at = datetime.now(timezone.utc).isoformat()
            summary = {**{key: plan[key] for key in (*ident, "backend_digest", "resources", "output", "source_run")},
                       "schema_version": 1, "baseline_type": "no_perturbation", "evaluation_type": "direct",
                       "status": "completed", "completed_at": completed_at, "updated_at": completed_at,
                       "matching": matching, "results": matching, "no_noise": plan["no_noise"],
                       "candidate_count": matching.get("candidate_count", 0),
                       "matched_count": matching.get("matched_count", 0), "match_rate": matching.get("match_rate", 0.0)}
            write_json(output / "summary.json", summary)
            archive_result(settings, "direct", summary, format_summary(summary, include_output=False))
            manifest["status"] = "completed"
            print_summary(summary)
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(output / "run_manifest.json", manifest)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"直接评估未完成：{exc}")
        return 1


# Public aliases keep the backend convenient for callers that do not need to
# know the historical ``run_direct`` name.
def run(settings: dict, model: str, finetune: str, *, mode: str = "run", dataset: str | None = None, source_run: str | None = None) -> int:
    return run_direct(settings, model, finetune, mode=mode, dataset=dataset, source_run=source_run)


def preview(settings: dict, model: str, finetune: str, *, dataset: str | None = None) -> int:
    return run_direct(settings, model, finetune, mode="preview", dataset=dataset)


def history(settings: dict, model: str, finetune: str, *, dataset: str | None = None) -> int:
    return run_direct(settings, model, finetune, mode="history", dataset=dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("chatglm", "chatglm3", "fm9g"))
    parser.add_argument("--finetune", choices=("explicit", "implicit"))
    parser.add_argument("--dataset", choices=("sst2", "wikitext2", "wikitext"))
    parser.add_argument("--mode", choices=("run", "preview", "history", "evaluate"), default="preview")
    parser.add_argument("--source-run")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--match", action="store_true")
    parser.add_argument("--generated")
    parser.add_argument("--train-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    parser.add_argument("--contains-min-chars", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--window-min-tokens", type=int, default=8)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.match:
        if not all((args.generated, args.train_data, args.output_dir, args.dataset)):
            raise SystemExit("--match 需要 --dataset、--generated、--train-data 和 --output-dir")
        return run_match(args)
    if not args.model or not args.finetune:
        raise SystemExit("需要 --model 和 --finetune")
    return run_direct(load_settings(args.config, args.set), args.model, args.finetune, mode=args.mode, dataset=args.dataset, source_run=args.source_run)


if __name__ == "__main__":
    raise SystemExit(main())
