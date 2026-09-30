"""Shared evaluate/ integration; no ML imports in the menu process."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from uuid import uuid4

from .core import PROJECT_ROOT, configured_python, get_nested, load_settings, project_path
from .evaluation_history import archive_result, load_result

ENGINE = "evaluate-indirect-v1"
ROOTS = {"explicit": PROJECT_ROOT / "evaluate/indirect_LoRA",
         "implicit": PROJECT_ROOT / "evaluate/indirect_ICL"}
MODEL_INFO = {"chatglm3": ("CHATGLM_MODEL_DIR", "chatglm3-6b", "chatglm3_6b"),
              "fm9g": ("FM9G_MODEL_DIR", "9G7B_MHA", "fm9g_7b")}
FIELDS = ("original_text",)
TARGETS = (.01, .001, .0001)
TARGET_LABELS = ((.01, "1%"), (.001, "0.1%"), (.0001, "0.01%"))
FINETUNE_LABELS = {"explicit": "显式", "implicit": "隐式"}


def path(value):
    return project_path(str(value)).resolve()


def identity(model, kind, dataset):
    model = "chatglm3" if model == "chatglm" else model
    if model not in MODEL_INFO or kind not in ROOTS or dataset not in {"sst2", "wikitext2"}:
        raise ValueError(f"不支持的间接评估组合：{model}/{kind}/{dataset}")
    return dict(model=model, finetune_type=kind, dataset=dataset, engine=ENGINE)


def backend_digest(kind):
    root = ROOTS[kind]
    files = [root / "run.sh"]
    if kind == "explicit":
        files += [root / "run.py", *sorted((root / "code").glob("*.py"))]
    else:
        files += sorted((root / "DPICL").rglob("*.py"))
    files += [Path(__file__), Path(__file__).with_name("indirect_data_check.py")]
    h = hashlib.sha256()
    for file in files:
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def result_root(settings, ident):
    return path(settings.get("results_dir", "privacy_tools/results")) / "indirect_evaluation" / ident["model"] / ident["finetune_type"] / ident["dataset"]


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write_json(file, value):
    Path(file).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_plan(settings, model, kind, dataset, output, *, source_run=None):
    """Pure command construction. Neither creates directories nor imports torch."""
    ident = identity(model, kind, dataset)
    model = ident["model"]
    root, output = ROOTS[kind], path(output)
    backend = output / "backend"
    opts = get_nested(settings, "evaluation.indirect", {})
    if not isinstance(opts, dict):
        raise ValueError("evaluation.indirect 必须为 JSON 对象")
    def option(key, default=None):
        return get_nested(opts, key, default)
    py = configured_python(settings, "indirect_lora" if kind == "explicit" else "indirect_icl")
    env_key, folder, common_key = MODEL_INFO[model]
    configured = option(f"models.{model}.base_model") or get_nested(settings, f"common.{common_key}")
    # Old hub identifiers are not silently reinterpreted as repository paths.
    base_value = os.environ.get(env_key) or configured or str(root.parent / "models" / folder)
    base = path(base_value)
    warnings = []
    if str(base_value).startswith(("THUDM/", "OpenBMB/")):
        warnings.append(f"{env_key} 或 evaluation.indirect.models.{model}.base_model 必须指定本地模型目录，当前值为 {base_value}")
    gpu_text = str(get_nested(settings, "common.gpu", "0"))
    if source_run is None and gpu_text not in {"0", "1", "2", "3"}:
        raise ValueError("新版间接评估要求 common.gpu 为单个物理卡号 0/1/2/3")
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg"}
    resources = {}
    stages = []
    if source_run is not None:
        source = path(source_run)
        summary = read_json(source / "summary.json")
        if any(summary.get(k) != v for k, v in ident.items()) or summary.get("status") != "completed":
            raise ValueError("离线重算来源必须是模型、微调方式和数据集匹配的新版成功运行目录")
        if summary.get("backend_digest") != backend_digest(kind):
            raise ValueError("离线来源的后端版本不同；请显式迁移并核对实验口径")
        raw = source / "backend"
        if kind == "explicit":
            stages = [("evaluate", [py, str(root / "run.py"), "--model", model, "--dataset", dataset,
                                   "--mode", "evaluate", "--scores-dir", str(raw), "--output", str(backend)])]
            for stem in ("member", "nonmember"):
                if not (raw / f"{stem}_scores.pkl").is_file():
                    raise ValueError(f"缺少原始分数：{raw / (stem + '_scores.pkl')}")
        else:
            for field in FIELDS:
                dest = backend / field
                files = [raw / field / f"attack_results_{split}_bert_0_{field}.csv" for split in ("other", dataset)]
                for file in files:
                    if not file.is_file():
                        raise ValueError(f"缺少攻击 CSV：{file}")
                stages.extend([
                    (field + "_combine", [py, str(root / "DPICL/Indirect_attack/combine.py"),
                      "--member-file", str(files[0]), "--nonmember-file", str(files[1]), "--output", str(dest / "combined.csv")]),
                    (field + "_evaluate", [py, str(root / "DPICL/Indirect_attack/evaluate_attack_metrics.py"),
                      "--file", str(dest / "combined.csv"), "--output-dir", str(dest / "evaluation")])])
        resources = summary.get("resources", {})
    else:
        env.update({env_key: str(base), "CUDA_VISIBLE_DEVICES": gpu_text,
                    "PYTHONHASHSEED": "42" if kind == "explicit" else "0"})
        data = path(get_nested(settings, "common.dataset_path", "privacy_tools/datasets/sst2"))
        resources = dict(base_model=str(base), physical_gpu=int(gpu_text), python=py)
        if kind == "explicit":
            legacy = get_nested(settings, "methods.chatglm_lora.adapter_path") if model == "chatglm3" and dataset == "sst2" else None
            adapter = path(option(f"models.{model}.adapters.{dataset}") or legacy or root / "weights" / model / dataset)
            tokenizer = path(option(f"models.{model}.tokenizers.{dataset}") or adapter)
            neighbor = path(option("lora.neighbor_model") or get_nested(settings, "methods.chatglm_lora.neighbor_model_path") or "/home/zjs/llm/roberta-base")
            training = option(f"models.{model}.training_manifests.{dataset}")
            resources.update(adapter=str(adapter), tokenizer=str(tokenizer), neighbor_model=str(neighbor),
                             training_manifest=str(path(training)) if training else str(adapter / "training_manifest.json"))
            args = ["--model", model, "--dataset", dataset, "--mode", "attack", "--gpu", gpu_text,
                    "--adapter-dir", str(adapter), "--tokenizer-dir", str(tokenizer),
                    "--neighbor-model-path", str(neighbor), "--output", str(backend)]
            if dataset == "sst2":
                resources.update(member_file=str(data / "train.tsv"), nonmember_file=str(data / "test.tsv"))
                args += ["--member-data-file", resources["member_file"], "--nonmember-data-file", resources["nonmember_file"]]
            else:
                resources["dataset_dir"] = str(path(option("datasets.wikitext2.lora_dir") or root / "data/wikitext2"))
                args += ["--dataset-dir", resources["dataset_dir"]]
            stages.append(("lora", [py, str(root / "run.py"), *args]))
        else:
            proposal = option("icl.proposal_model")
            old = get_nested(settings, "common.icl_proposal_model")
            if old and not proposal and "roberta" in str(old).lower():
                warnings.append("旧 ICL RoBERTa 配置不再继承，使用新版 BERT；可设置 evaluation.indirect.icl.proposal_model")
            proposal = path(proposal or (old if old and "roberta" not in str(old).lower() else None) or root / "resources/bert-base-uncased")
            resources["neighbor_model"] = str(proposal)
            args = ["--target-model", model, "--generation-model", model, "--proposal-model", str(proposal),
                    "--gpu", "0", "--trust-remote-code", "--run-dir", str(backend), "--task-dataset", dataset,
                    "--num-nonmembers", "500", "--num-demonstrations", "10", "--seed", "0",
                    "--score-mode", "token-mean", "--dtype", "float32", "--generation-dtype", "float16",
                    "--max-token-per-text", "100", "--prompt-start", "5", "--prompt-stop", "10"]
            if dataset == "sst2":
                resources["dataset_dir"] = str(data)
                args += ["--sst2-dir", str(data), "--num-test", "20", "--max-length", "512"]
            else:
                resources["dataset_dir"] = str(path(option("datasets.wikitext2.icl_dir") or root / "data/WikiText-2-raw-v1"))
                args += ["--wikitext2-dir", resources["dataset_dir"], "--wikitext2-config", "wikitext-2-raw-v1",
                         "--num-test", "0", "--max-length", "2048", "--proposal-max-length", "512"]
            input_json = option("icl.input_json")
            if input_json:
                resources["input_json"] = str(path(input_json))
                args += ["--input-json", resources["input_json"]]
            stages.append(("icl", [py, str(root / "DPICL/scripts/run_indirect.py"), *args]))
    return {**ident, "backend_digest": backend_digest(kind), "output": str(output),
            "backend": str(backend), "python": py, "env": env, "resources": resources,
            "warnings": warnings, "stages": stages, "source_run": str(path(source_run)) if source_run else None}


def require_model(folder, *, adapter=False, tokenizer=False):
    folder = Path(folder)
    if tokenizer:
        if not (folder / "tokenizer_config.json").is_file() or not any((folder / n).is_file() for n in ("tokenizer.json", "tokenizer.model", "vocab.txt", "vocab.json")):
            raise ValueError(f"缺少 tokenizer 文件：{folder}")
        return
    config = folder / ("adapter_config.json" if adapter else "config.json")
    if not config.is_file():
        raise ValueError(f"缺少模型配置：{config}")
    if adapter:
        names = [folder / "adapter_model.safetensors"]
    else:
        names = []
        for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (folder / index).is_file():
                names = [folder / name for name in set(read_json(folder / index)["weight_map"].values())]
                break
        if not names:
            names = [next((folder / n for n in ("model.safetensors", "pytorch_model.bin") if (folder / n).is_file()), folder / "model.safetensors")]
    if any(not n.is_file() or n.stat().st_size == 0 for n in names):
        raise ValueError(f"缺少或为空的模型权重：{folder}")


def preflight(plan):
    if not shutil.which(plan["python"]):
        raise ValueError(f"Python 不可用：{plan['python']}；配置 python.indirect_lora / python.indirect_icl")
    if plan["source_run"]:
        return
    r = plan["resources"]
    if any("必须指定本地模型" in w for w in plan["warnings"]):
        raise ValueError("; ".join(plan["warnings"]))
    require_model(r["base_model"])
    require_model(r["base_model"], tokenizer=True)
    require_model(r["neighbor_model"])
    require_model(r["neighbor_model"], tokenizer=True)
    if plan["finetune_type"] == "explicit":
        require_model(r["adapter"], adapter=True)
        require_model(r["tokenizer"], tokenizer=True)
    elif read_json(Path(r["neighbor_model"]) / "config.json").get("model_type") != "bert":
        raise ValueError("ICL proposal 必须使用 BERT-base-uncased 兼容资源，不能使用旧 RoBERTa 配置")


def finite_number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("指标出现 NaN/Inf")
    return value


def collect_results(plan):
    root = Path(plan["backend"])
    results = {}
    if plan["finetune_type"] == "explicit":
        file = root / "mia_evaluation_results.csv"
        with file.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(TARGETS) or {float(r["target_fpr"]) for r in rows} != set(TARGETS):
            raise ValueError("LoRA 指标缺少目标 FPR 行")
        for row in rows:
            for key in ("target_fpr", "actual_fpr", "tpr", "auc", "member_count", "nonmember_count", "tp", "fp", "tn", "fn"):
                row[key] = finite_number(row[key])
            if row["actual_fpr"] > row["target_fpr"] or any(not 0 <= row[k] <= 1 for k in ("auc", "tpr", "actual_fpr")):
                raise ValueError("LoRA 输出指标不合法")
            counts = ("tp", "fp", "tn", "fn", "member_count", "nonmember_count")
            if any(row[k] < 0 or not row[k].is_integer() for k in counts):
                raise ValueError("LoRA 样本与混淆矩阵计数必须为非负整数")
            if (row["tp"] + row["fn"] != row["member_count"] or
                    row["fp"] + row["tn"] != row["nonmember_count"]):
                raise ValueError("LoRA 混淆矩阵与实际样本数不一致")
        first = rows[0]
        results["lora"] = dict(auc=first["auc"], member_count=int(first["member_count"]),
                               nonmember_count=int(first["nonmember_count"]), rows=rows, metrics_file=str(file))
        if any(r["member_count"] != 1200 or r["nonmember_count"] != 1200 or r["auc"] != first["auc"] for r in rows):
            raise ValueError("LoRA 正式结果应有 1200+1200 条有效分数且 AUC 一致")
    else:
        for field in FIELDS:
            file = root / field / "evaluation/metrics.json"
            metrics = read_json(file)
            for key in ("AUC", "TPR@1%", "TPR@0.1%", "TPR@0.01%"):
                if not 0 <= finite_number(metrics[key]) <= 1:
                    raise ValueError(f"ICL 指标越界：{key}")
            if metrics["member_count"] != 10 or metrics["nonmember_count"] != 500:
                raise ValueError("ICL 正式结果应有 10+500 条有效分数")
            results[field] = {**metrics, "metrics_file": str(file)}
    return results


def format_summary(summary, *, include_output=True):
    finetune = FINETUNE_LABELS.get(summary["finetune_type"], summary["finetune_type"])
    lines = [f"方法：间接评估 模型：{summary['model']}，微调方式：{finetune}，数据集：{summary['dataset']}"]
    for field, metrics in summary["results"].items():
        lines.append(f"{field}: AUC={metrics.get('auc', metrics.get('AUC')):.6f}, member={metrics['member_count']}, nonmember={metrics['nonmember_count']}")
        if "rows" in metrics:
            rows = {float(row["target_fpr"]): row for row in metrics["rows"]}
            for target, label in TARGET_LABELS:
                if target in rows:
                    lines.append(f"TPR@{label}FPR: {rows[target]['tpr']:.2%}")
        else:
            for _, label in TARGET_LABELS:
                key = f"TPR@{label}"
                if key in metrics:
                    lines.append(f"TPR@{label}FPR: {metrics[key]:.2%}")
    if include_output:
        lines.append(f"结果目录：{summary['output']}")
    return "\n".join(lines)


def print_summary(summary):
    print(format_summary(summary))


def show_history(settings, ident):
    history = load_result(settings, "indirect", ident)
    if history is None:
        print("未找到当前模型、微调方式和数据集的无扰动历史结果。")
        return 0
    if history["kind"] == "text":
        print(history["text"], end="")
    else:
        print_summary(history["summary"])
    return 0


def run_indirect(settings, model, kind, *, mode="run", dataset=None, source_run=None):
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
            print("命令预览（不启动计算、不创建目录；资源与成员来源尚未验收）")
            print("环境：" + shlex.join([f"{k}={v}" for k, v in plan["env"].items()]))
            # Expand LoRA's native stages without executing a subprocess.
            if kind == "explicit":
                import importlib.util
                spec = importlib.util.spec_from_file_location("_evaluate_lora_entry", ROOTS[kind] / "run.py")
                native = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(native)
                native.model_path = lambda model: Path(plan["resources"]["base_model"])
                r = plan["resources"]
                for split, key in (("train", "member_file"), ("nonmember", "nonmember_file")):
                    cmd = native.attack_command(ident["model"], ident["dataset"], split, Path(plan["backend"]),
                          adapter=r["adapter"], tokenizer=r["tokenizer"], neighbor_model=r["neighbor_model"],
                          dataset_file=r.get(key), dataset_dir=r.get("dataset_dir"))
                    cmd[0] = plan["python"]
                    print(shlex.join(cmd))
                cmd = native.evaluation_command(Path(plan["backend"]), Path(plan["backend"]))
                cmd[0] = plan["python"]
                print(shlex.join(cmd))
            else:
                for name, cmd in plan["stages"]:
                    print(name + ": " + shlex.join(cmd) + " --dry-run")
            return 0
        preflight(plan)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {**plan, "status": "running", "stage_results": []}
        write_json(output / "run_manifest.json", manifest)
        env = {**os.environ, **plan["env"]}
        env.setdefault("PYTHONUNBUFFERED", "1")
        stages = list(plan["stages"])
        # TODO: Re-enable data provenance validation before formal evaluation.
        # if not source_run:
        #     stages.insert(0, ("preflight", [plan["python"], str(PROJECT_ROOT / "privacy_tools/indirect_data_check.py"),
        #                                   "--manifest", str(output / "run_manifest.json")]))
        try:
            for name, cmd in stages:
                start = time.monotonic()
                print(f"START {name}; 日志：{output / (name + '.log')}", flush=True)
                with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                manifest["stage_results"].append(dict(name=name, returncode=returncode, seconds=time.monotonic()-start))
                write_json(output / "run_manifest.json", manifest)
                if returncode:
                    raise RuntimeError(f"{name} 失败（退出码 {returncode}），查看 {output / (name + '.log')}")
            results = collect_results(plan)
            receipt = (read_json(output / "data_validation.json")
                       if not source_run and (output / "data_validation.json").is_file()
                       else {"status": "skipped"} if not source_run
                       else read_json(path(source_run) / "summary.json").get("data_validation"))
            completed_at = datetime.now(timezone.utc).isoformat()
            summary = {**{k: plan[k] for k in (*ident, "backend_digest", "resources", "output", "source_run")},
                       "schema_version": 1, "baseline_type": "no_perturbation", "evaluation_type": "indirect",
                       "status": "completed", "completed_at": completed_at, "updated_at": completed_at,
                       "results": results, "data_validation": receipt}
            write_json(output / "summary.json", summary)
            archive_result(settings, "indirect", summary, format_summary(summary, include_output=False))
            manifest["status"] = "completed"
            print_summary(summary)
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(output / "run_manifest.json", manifest)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"间接评估未完成：{exc}")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("chatglm", "chatglm3", "fm9g"), required=True)
    parser.add_argument("--finetune", choices=ROOTS, required=True)
    parser.add_argument("--dataset", choices=("sst2", "wikitext2"))
    parser.add_argument("--mode", choices=("run", "preview", "history", "evaluate"), default="preview")
    parser.add_argument("--source-run")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    return run_indirect(load_settings(args.config, args.set), args.model, args.finetune,
                        mode=args.mode, dataset=args.dataset, source_run=args.source_run)


if __name__ == "__main__":
    raise SystemExit(main())
