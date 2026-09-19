from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parent
DEFAULT_SETTINGS_FILE = TOOL_ROOT / "settings.example.json"


@dataclass
class Step:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    source_script: Path | None = None
    replacements: dict[str, str] = field(default_factory=dict)


@dataclass
class Experiment:
    id: str
    title: str
    description: str
    builder: Any | None = None
    internal_report: bool = False
    expensive: bool = True
    kind: str = "experiment"


@dataclass
class Method:
    id: str
    title: str
    description: str
    root: Path
    required_files: list[Path]
    experiments: list[Experiment]
    warnings: list[str] = field(default_factory=list)
    path_name: str | None = None


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(settings: dict[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"参数覆盖必须采用 key=value：{item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = settings
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"无法覆盖非对象配置：{dotted_key}")
            cursor = child
        cursor[keys[-1]] = parse_value(raw_value)


def load_settings(config_file: str | None, overrides: Iterable[str]) -> dict[str, Any]:
    settings = json.loads(DEFAULT_SETTINGS_FILE.read_text(encoding="utf-8"))
    if config_file:
        path = Path(config_file).expanduser().resolve()
        settings = deep_merge(settings, json.loads(path.read_text(encoding="utf-8")))
        settings["_config_file"] = str(path)
    apply_overrides(settings, overrides)
    return settings


def get_nested(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def configured_python(settings: dict[str, Any], method_id: str) -> str:
    return str(get_nested(settings, f"python.{method_id}", get_nested(settings, "python.default", "python3")))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def python_step(
    settings: dict[str, Any],
    method_id: str,
    name: str,
    script: Path,
    args: list[Any],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    replacements: dict[str, str] | None = None,
    python_profile: str | None = None,
) -> Step:
    command = [configured_python(settings, python_profile or method_id), str(script), *[str(arg) for arg in args]]
    return Step(
        name=name,
        command=command,
        cwd=cwd,
        env=env or {},
        source_script=script if replacements else None,
        replacements=replacements or {},
    )


def method_env(settings: dict[str, Any], *python_paths: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    gpu = str(get_nested(settings, "common.gpu", "0"))
    env["CUDA_VISIBLE_DEVICES"] = gpu
    endpoint = get_nested(settings, "common.hf_endpoint")
    if endpoint:
        env["HF_ENDPOINT"] = str(endpoint)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(path) for path in python_paths]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # 攻击评估脚本会绘图；命令行/服务器环境使用无界面后端，避免 plt.show() 阻塞。
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def safe_console_write(value: str) -> None:
    """在 Windows GBK 控制台也能展示任意 UTF-8 子进程输出。"""
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(value.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    sys.stdout.flush()


def evaluation_scope(step_name: str) -> str:
    """Return a user-facing protected/baseline label for an evaluation step."""
    name = step_name.casefold()
    comparison_markers = ("比较受保护与未保护", "比较受保护与非保护", "对比受保护", "保护与非保护")
    baseline_markers = ("未保护", "非保护", "对比方法", "原始提示", "原始示例")
    protected_markers = ("受保护", "所提出方法", "合成提示", "合成示例", "privsv")
    if any(marker in name for marker in comparison_markers):
        return "保护方法 vs 非保护/对比方法"
    if any(marker in name for marker in baseline_markers):
        return "非保护/对比方法"
    if any(marker in name for marker in protected_markers):
        return "保护方法"
    return "当前方法评估流程"


def evaluation_banner(method: Method, experiment: Experiment, step: Step, display_title: str | None = None) -> str | None:
    if experiment.id not in {"direct_attack", "indirect_attack"}:
        return None
    return f"当前评估：{display_title or method.title}｜{evaluation_scope(step.name)}"


def _materialize_patched_step(step: Step, run_dir: Path, index: int) -> Step:
    if step.source_script is None:
        return step
    text = step.source_script.read_text(encoding="utf-8")
    missing: list[str] = []
    for old, new in step.replacements.items():
        if old not in text:
            missing.append(old)
        text = text.replace(old, new)
    if missing:
        missing_preview = ", ".join(repr(item[:80]) for item in missing)
        raise ValueError(f"运行时适配失败，原脚本中未找到待替换内容：{missing_preview}")
    runtime_dir = run_dir / "runtime_adapters"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    patched = runtime_dir / f"{index:02d}_{step.source_script.name}"
    patched.write_text(text, encoding="utf-8")
    command = list(step.command)
    command[1] = str(patched)
    return Step(step.name, command, step.cwd, step.env)


def printable_command(step: Step) -> str:
    try:
        return shlex.join(step.command)
    except AttributeError:
        return " ".join(shlex.quote(part) for part in step.command)


PRUNED_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    "embeddings",
    "models",
    "wandb",
    "node_modules",
}

INTERESTING_NAMES = {
    "attack_evaluation.json",
    "attack_generation.json",
    "shared_legacy_attack_evaluation.json",
    "native_indirect_attack.json",
    "native_indirect_attack.csv",
    "cliptext_outputs.json",
    "benchmark.json",
    "evaluation_benchmark.json",
    "eval_metrics.json",
    "all_results.json",
    "overall_stats.json",
    "trainer_state.json",
    "training_metrics.csv",
    "dp_training_metrics.csv",
    "experiment_logs.jsonl",
    "mia_fair_comparison_results.csv",
    "groups.csv",
    "runs.csv",
    "synthetic_text_results.json",
}


def discover_artifacts(root: Path, since: float | None = None, limit: int = 400) -> list[Path]:
    found: list[Path] = []
    if not root.exists():
        return found
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in PRUNED_DIRS]
        current_path = Path(current)
        for name in filenames:
            lowered = name.lower()
            interesting = (
                name in INTERESTING_NAMES
                or lowered.startswith("run_metrics") and lowered.endswith(".json")
                or lowered.startswith("attack_results") and lowered.endswith(".csv")
                or lowered.endswith(".log")
            )
            if not interesting:
                continue
            path = current_path / name
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > 8 * 1024 * 1024:
                continue
            if since is not None and stat.st_mtime + 1 < since:
                continue
            found.append(path)
            if len(found) >= limit:
                return sorted(found)
    return sorted(found)


def _flatten_numbers(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numbers(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value[-20:]):
            rows.extend(_flatten_numbers(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append((prefix, float(value)))
    return rows


def parse_structured_file(path: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in _flatten_numbers(data):
                observations.append({"metric": key, "value": value, "source": str(path)})
        elif path.suffix.lower() == ".jsonl":
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context: list[str] = []
                if isinstance(data, dict):
                    for context_key in ("steering_type", "stage"):
                        context_value = data.get(context_key)
                        if isinstance(context_value, str) and context_value:
                            context.append(re.sub(r"[^a-zA-Z0-9_-]+", "_", context_value).strip("_").lower())
                    multiplier = data.get("multiplier")
                    if isinstance(multiplier, (int, float)):
                        context.append(f"multiplier_{multiplier:g}")
                prefix = ".".join(context)
                for key, value in _flatten_numbers(data, prefix):
                    observations.append({"metric": key, "value": value, "source": f"{path}#tail{line_number}"})
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                rows = list(csv.DictReader(handle))[-100:]
            for row_number, row in enumerate(rows, 1):
                for key, raw in row.items():
                    if raw is None:
                        continue
                    values = raw if isinstance(raw, list) else [raw]
                    for value_index, item in enumerate(values, 1):
                        if not isinstance(item, (str, int, float)):
                            continue
                        cleaned = str(item).strip().rstrip("%")
                        try:
                            value = float(cleaned)
                        except ValueError:
                            continue
                        metric = key or (f"unnamed_{value_index}" if len(values) > 1 else "unnamed")
                        observations.append({"metric": metric, "value": value, "source": f"{path}#tail{row_number}"})
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
        return []
    return observations


def print_historical_results(
    root: Path,
    method_id: str,
    method_title: str | None = None,
    experiment_id: str | None = None,
    real_execution: bool = False,
    display_title: str | None = None,
    baseline_only: bool = False,
) -> None:
    """Print the prepared result for the selected method and experiment kind."""
    if not root.exists():
        print(f"\n历史结果目录不存在，跳过历史结果展示：{root}")
        return
    filenames = {
        "cliptext": "ClipText.txt",
        "ptcds": "PTCDS.txt",
        "kal_dp": "Kal-DP.txt",
        "zo_dpga": "ZO-DPGA.txt",
        "dadp_icl": "DADP-ICL.txt",
        "privsv": "PrivSV.txt",
    }
    filename = filenames.get(method_id)
    if filename is None and method_title:
        filename = f"{method_title}.txt"
    if not filename:
        print(f"\n未找到方法 {method_id} 对应的历史结果文件。")
        return
    # Training/baseline performance files live at the history root. Attack
    # evaluations are separated into direct/indirect subdirectories.
    if experiment_id in {"direct_attack", "indirect_attack"}:
        result_file = root / experiment_id.replace("_attack", "") / filename
        result_label = "直接评估" if experiment_id == "direct_attack" else "间接评估"
    else:
        result_file = root / filename
        result_label = "性能"
    if not result_file.is_file():
        print(f"\n当前方法没有历史结果文件，跳过历史结果展示：{result_file}")
        return
    try:
        content = result_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        print(f"\n读取历史结果失败：{result_file}（{exc}）")
        return
    if baseline_only:
        match = re.search(r"(?ms)^基准方法\s*$.*?(?=^\S|\Z)", content)
        content = match.group(0).strip() if match else ""
    elif experiment_id == "baseline":
        lines = [line for line in content.splitlines() if re.search(r"准确率|Accuracy|基准方法|peft|Prompt|非隐私|对比方法", line, re.I)]
        content = "\n".join(lines)
    elif experiment_id == "performance_baseline":
        lines = [line for line in content.splitlines() if re.search(r"内存|memory|运行时间|runtime|耗时|吞吐率|throughput|samples?/sec|DP-SGD|性能对比", line, re.I)]
        content = "\n".join(lines)
    print(f"\n===== {display_title or method_title or method_id} {result_label}结果 =====")
    print(content or "（历史结果文件中没有匹配的指标）")
    if real_execution and experiment_id == "utility":
        print("\n其中基准方法结果为历史结果")
    elif real_execution and experiment_id in {"baseline", "performance_baseline"}:
        print("\n其中所提出方法结果为历史结果")


METRIC_PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    ("accuracy_pct", re.compile(r"(?:eval[_ /-]?accuracy|val[_ /-]?acc|accuracy|dev acc|\bacc)(?:\s*\([^)]*\))?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(%)?", re.I), True),
    ("epsilon", re.compile(r"(?:epsilon|eps|ε|ϵ)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I), False),
    ("reported_throughput_samples_per_second", re.compile(r"(?:avg\s+)?(?:throughput|samples_per_second|examples_per_second)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:samples?/s|examples?/s|smp/s)?", re.I), False),
    ("peak_memory_gb", re.compile(r"(?:peak[^\n]{0,30}memory|memory[^\n]{0,20}peak|\bmem)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*gb", re.I), False),
    ("peak_memory_mb", re.compile(r"(?:peak[^\n]{0,30}memory|memory[^\n]{0,20}peak|\bmem)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*mb", re.I), False),
    ("attack_success_pct", re.compile(r"(?:attack success|攻击成功率|成功率)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I), False),
    ("attack_success_pct", re.compile(r"占[^\n]{0,40}比例(?:为|是)?\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I), False),
    ("attack_success_pct", re.compile(r"整段命中[^\n]{0,40}\(([0-9]+(?:\.[0-9]+)?)\s*%\)", re.I), False),
    ("attack_auc", re.compile(r"\bAUC\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I), False),
    ("attack_tpr_at_1pct", re.compile(r"TPR@1%\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I), False),
    ("runtime_seconds", re.compile(r"(?:all done in|elapsed(?:_program)?|overall runtime(?:\s*\(s\))?|total time|training wall(?: time)?)\s*[=: ]+\s*([0-9]+(?:\.[0-9]+)?)\s*s?", re.I), False),
]


def parse_log_metrics(text: str, source: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for metric, pattern, normalize_accuracy in METRIC_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            if normalize_accuracy:
                has_percent = bool(match.lastindex and match.lastindex >= 2 and match.group(2))
                if not has_percent and value <= 1.0:
                    value *= 100.0
            observations.append({"metric": metric, "value": value, "source": source})
    return observations


def _numeric_observations(observations: list[dict[str, Any]]) -> list[tuple[str, float, str]]:
    rows: list[tuple[str, float, str]] = []
    for item in observations:
        value = item.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append((str(item.get("metric", "")).lower(), float(value), str(item.get("source", ""))))
    return rows


def _relative_source(source: str) -> str:
    base, marker, suffix = source.partition("#")
    try:
        shown = Path(base).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        shown = Path(base).name or base
    return f"{shown}#{suffix}" if marker else shown


def _accuracy_metric(rows: list[tuple[str, float, str]], method_id: str) -> tuple[float | None, str]:
    candidates: list[tuple[int, float, str]] = []
    for key, value, source in rows:
        if any(token in key for token in ("attack", "membership", "mia_")):
            continue
        privsv_result = method_id == "privsv" and key.endswith(".result") and "private_mean" in key and "multiplier_2" in key
        if not (privsv_result or "accuracy" in key or re.search(r"(?:^|[._\[])acc(?:$|[._\]])", key)):
            continue
        normalized = value * 100.0 if 0 <= value <= 1.0 else value
        if not 0 <= normalized <= 100:
            continue
        priority = 5 if privsv_result else 4 if "calibrated_w_dp" in key else 3 if any(token in key for token in ("eval", "test", "with_dp", "w_dp")) else 2
        candidates.append((priority, normalized, source))
    if not candidates:
        return None, ""
    best_priority = max(item[0] for item in candidates)
    _, value, source = [item for item in candidates if item[0] == best_priority][-1]
    return round(value, 6), _relative_source(source)


def _memory_metric(rows: list[tuple[str, float, str]]) -> tuple[float | None, str, str]:
    candidates: list[tuple[int, float, str, str]] = []
    for key, value, source in rows:
        if not any(token in key for token in ("memory", "mem", "peak_allocated_bytes", "peak_reserved_bytes")):
            continue
        if not any(token in key for token in ("peak", "max", "memory_gb", "memory_mb")):
            continue
        if "bytes" in key:
            value_gb = value / (1024 ** 3)
        elif re.search(r"(?:^|[._])mb(?:$|[._])", key) or "_mb" in key:
            value_gb = value / 1024
        else:
            value_gb = value
        if value_gb < 0:
            continue
        if any(token in key for token in ("cuda", "gpu")):
            priority, scope = 3, "GPU/CUDA"
        elif any(token in key for token in ("rss", "cpu")):
            priority, scope = 1, "进程 RSS/CPU"
        else:
            priority, scope = 2, "原实验报告值"
        candidates.append((priority, value_gb, source, scope))
    if not candidates:
        return None, "", ""
    best_priority = max(item[0] for item in candidates)
    _, value, source, scope = max((item for item in candidates if item[0] == best_priority), key=lambda item: item[1])
    return round(value, 6), _relative_source(source), scope


def _runtime_metric(rows: list[tuple[str, float, str]], fallback_seconds: float) -> tuple[float, str]:
    candidates: list[tuple[int, float, str]] = []
    for key, value, source in rows:
        if value < 0:
            continue
        priority = 0
        if any(token in key for token in ("overall_runtime_seconds", "total_elapsed_seconds")):
            priority = 5
        elif "training_wall_seconds" in key or "total_time" in key:
            priority = 4
        elif key == "runtime_seconds" or key.endswith(".runtime_seconds"):
            priority = 3
        elif "wall_seconds" in key or "elapsed_seconds" in key:
            priority = 2
        if priority:
            candidates.append((priority, value, source))
    if not candidates:
        return round(fallback_seconds / 3600, 6), "统一入口计时"
    best_priority = max(item[0] for item in candidates)
    _, value, source = max((item for item in candidates if item[0] == best_priority), key=lambda item: item[1])
    return round(value / 3600, 6), _relative_source(source)


def _throughput_metric(rows: list[tuple[str, float, str]]) -> tuple[float | None, str]:
    candidates: list[tuple[int, float, str]] = []
    for key, value, source in rows:
        if value < 0 or not any(token in key for token in ("throughput", "samples_per_second", "examples_per_second")):
            continue
        priority = 4 if any(token in key for token in ("overall", "avg", "private_samples", "train_samples_per_second")) else 3 if "reported" in key else 2
        candidates.append((priority, value, source))
    if not candidates:
        return None, ""
    best_priority = max(item[0] for item in candidates)
    selected = [item for item in candidates if item[0] == best_priority]
    value = sum(item[1] for item in selected) / len(selected)
    return round(value, 6), _relative_source(selected[-1][2])


def _epsilon_metric(rows: list[tuple[str, float, str]], method_id: str, experiment_id: str, settings: dict[str, Any]) -> tuple[float | None, str]:
    candidates: list[tuple[float, str]] = []
    for key, value, source in rows:
        if value < 0 or any(token in key for token in ("zo_eps", "empirical", "epsilon_tag")):
            continue
        if key in {"epsilon", "eps"} or key.endswith((".epsilon", ".dp_epsilon", "_dp_epsilon")):
            candidates.append((value, source))
    if candidates:
        value, source = candidates[-1]
        return round(value, 8), _relative_source(source)
    configured_budget_experiments = {
        "cliptext": {"utility"},
        "ptcds": {"utility"},
        "kal_dp": {"utility", "baseline"},
    }
    if experiment_id in configured_budget_experiments.get(method_id, set()):
        value = get_nested(settings, f"methods.{method_id}.epsilon", None)
        if isinstance(value, (int, float)):
            return round(float(value), 8), "本次运行参数"
    return None, ""


def _privacy_parameter(method_id: str, experiment_id: str, settings: dict[str, Any]) -> tuple[str, Any]:
    utility_experiments = {
        "cliptext": {"utility"},
        "ptcds": {"utility"},
        "kal_dp": {"utility", "baseline"},
        "zo_dpga": {"utility", "baseline"},
        "dadp_icl": {"utility", "baseline"},
        "privsv": {"utility", "baseline"},
    }
    if experiment_id not in utility_experiments.get(method_id, set()):
        return "", ""
    if method_id == "dadp_icl":
        value = 0 if experiment_id == "baseline" else get_nested(settings, "methods.dadp_icl.sigma", "")
        return "sigma", value
    if method_id == "privsv":
        value = 0 if experiment_id == "baseline" else get_nested(settings, "methods.privsv.noise_multiplier", "")
        return "noise_multiplier", value
    return "epsilon", get_nested(settings, f"methods.{method_id}.epsilon", "")


METRICS_FIELDS = [
    "run_id",
    "finished_at",
    "method",
    "experiment",
    "privacy_budget_epsilon",
    "privacy_parameter_name",
    "privacy_parameter_value",
    "accuracy_pct",
    "peak_memory_gb",
    "memory_scope",
    "runtime_hours",
    "throughput_samples_per_second",
    "status",
    "accuracy_source",
    "memory_source",
    "runtime_source",
    "throughput_source",
    "epsilon_source",
]


def write_metrics_result(
    run_dir: Path,
    method: Method,
    experiment: Experiment,
    return_code: int,
    started_at: float,
    artifacts: list[Path],
    settings: dict[str, Any],
) -> Path:
    observations: list[dict[str, Any]] = []
    stdout_file = run_dir / "stdout.log"
    if stdout_file.exists():
        observations.extend(parse_log_metrics(stdout_file.read_text(encoding="utf-8", errors="replace"), str(stdout_file)))
    for artifact in artifacts:
        if artifact.resolve() == stdout_file.resolve():
            continue
        observations.extend(parse_structured_file(artifact))
        if artifact.suffix.lower() == ".log":
            try:
                observations.extend(parse_log_metrics(artifact.read_text(encoding="utf-8", errors="replace"), str(artifact)))
            except OSError:
                pass
    rows = _numeric_observations(observations)
    accuracy, accuracy_source = _accuracy_metric(rows, method.id)
    memory, memory_source, memory_scope = _memory_metric(rows)
    runtime, runtime_source = _runtime_metric(rows, time.time() - started_at)
    throughput, throughput_source = _throughput_metric(rows)
    epsilon, epsilon_source = _epsilon_metric(rows, method.id, experiment.id, settings)
    privacy_name, privacy_value = _privacy_parameter(method.id, experiment.id, settings)
    result = {
        "run_id": run_dir.name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "method": method.title,
        "experiment": experiment.title,
        "privacy_budget_epsilon": "" if epsilon is None else epsilon,
        "privacy_parameter_name": privacy_name,
        "privacy_parameter_value": privacy_value,
        "accuracy_pct": "" if accuracy is None else accuracy,
        "peak_memory_gb": "" if memory is None else memory,
        "memory_scope": memory_scope,
        "runtime_hours": runtime,
        "throughput_samples_per_second": "" if throughput is None else throughput,
        "status": "success" if return_code == 0 else f"failed({return_code})",
        "accuracy_source": accuracy_source,
        "memory_source": memory_source,
        "runtime_source": runtime_source,
        "throughput_source": throughput_source,
        "epsilon_source": epsilon_source,
    }
    if experiment.id == "baseline":
        for key in (
            "peak_memory_gb", "memory_scope", "runtime_hours",
            "throughput_samples_per_second", "memory_source",
            "runtime_source", "throughput_source",
        ):
            result[key] = ""
    elif experiment.id == "performance_baseline":
        result["accuracy_pct"] = ""
        result["accuracy_source"] = ""
    result_path = run_dir.parent / "experiment_metrics.csv"
    exists = result_path.exists() and result_path.stat().st_size > 0
    encoding = "utf-8" if exists else "utf-8-sig"
    with result_path.open("a" if exists else "w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(result)
    return result_path


def create_run_dir(settings: dict[str, Any], category: str, method_path_name: str, experiment_id: str) -> Path:
    result_root = project_path(str(settings.get("results_dir", "privacy_tools/results")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / category / method_path_name / experiment_id / f"{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_experiment(
    category: str,
    method: Method,
    experiment: Experiment,
    settings: dict[str, Any],
    *,
    dry_run: bool,
    display_title: str | None = None,
    historical_baseline_only: bool = False,
) -> int:
    method_path_name = method.path_name or method.title
    run_dir = create_run_dir(settings, category, method_path_name, experiment.id)
    started_at = time.time()
    manifest = {
        "schema_version": 1,
        "category": category,
        "method": method.id,
        "method_path_name": method_path_name,
        "experiment": experiment.id,
        "choice_kind": experiment.kind,
        "dry_run": dry_run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "config_file": settings.get("_config_file"),
        "steps": [],
    }
    materialized_steps: list[Step] = []
    try:
        steps: list[Step] = [] if experiment.internal_report else experiment.builder(settings, run_dir)
        for index, step in enumerate(steps, 1):
            actual = _materialize_patched_step(step, run_dir, index)
            matplotlib_dir = run_dir / ".matplotlib"
            matplotlib_dir.mkdir(parents=True, exist_ok=True)
            actual.env.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
            materialized_steps.append(actual)
            manifest["steps"].append({
                "name": actual.name,
                "cwd": str(actual.cwd),
                "command": actual.command,
                "environment_overrides": actual.env,
                "runtime_adapter": str(step.source_script) if step.source_script else None,
            })
    except Exception as exc:
        (run_dir / "stdout.log").write_text(f"准备实验失败：{exc}\n", encoding="utf-8")
        manifest["prepare_error"] = str(exc)
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"准备实验失败：{exc}", file=sys.stderr)
        print(f"运行记录：{run_dir}")
        if not dry_run:
            print("运行异常，请重新运行")
        return 2

    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n运行目录：{run_dir}")
    for index, step in enumerate(materialized_steps, 1):
        print(f"[{index}/{len(materialized_steps)}] {step.name}")
        banner = evaluation_banner(method, experiment, step, display_title)
        # if banner:
        #     print(f"  >>> {banner}")
        print(f"  cwd: {step.cwd}")
        print(f"  cmd: {printable_command(step)}")

    return_code = 0
    log_path = run_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        if dry_run:
            log.write("DRY RUN：未启动原实验代码。\n")
            for step in materialized_steps:
                log.write(f"[{step.name}] cwd={step.cwd}\n{printable_command(step)}\n")
        else:
            for index, step in enumerate(materialized_steps, 1):
                banner = evaluation_banner(method, experiment, step, display_title)
                if banner:
                    banner_line = f"\n>>> {banner}\n"
                    safe_console_write(banner_line)
                    log.write(banner_line)
                log.write(f"\n===== STEP {index}: {step.name} =====\n")
                log.flush()
                env = os.environ.copy()
                env.update(step.env)
                process: subprocess.Popen[str] | None = None
                try:
                    process = subprocess.Popen(
                        step.command,
                        cwd=step.cwd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        safe_console_write(line)
                        log.write(line)
                    return_code = process.wait()
                except KeyboardInterrupt:
                    if process is not None:
                        process.terminate()
                    return_code = 130
                    log.write("\n用户中断实验。\n")
                except OSError as exc:
                    return_code = 127
                    message = f"无法启动命令：{exc}\n"
                    print(message, file=sys.stderr)
                    log.write(message)
                log.flush()
                if return_code != 0:
                    break

    if not dry_run and return_code != 0:
        print("\n运行异常，请重新运行")
        return return_code

    history_root = project_path(str(settings.get("historical_results_dir", "privacy_tools/historical_results")))
    print_historical_results(
        history_root,
        method.id,
        method.title,
        experiment.id,
        real_execution=not dry_run,
        display_title=display_title,
        baseline_only=historical_baseline_only,
    )
    artifact_since = None if experiment.internal_report else started_at
    artifacts = discover_artifacts(method.root, since=artifact_since)
    if experiment.internal_report:
        integration_root = project_path(str(settings.get("results_dir", "privacy_tools/results"))) / category / method_path_name
        artifacts.extend(path for path in discover_artifacts(integration_root) if path not in artifacts)
    artifacts.extend(path for path in discover_artifacts(run_dir) if path not in artifacts)
    if dry_run:
        print("\n安全预演完成，当前输出结果为历史运行结果。")
    elif experiment.id in {"utility", "baseline", "performance_baseline"}:
        result_path = write_metrics_result(run_dir, method, experiment, return_code, started_at, artifacts, settings)
        print(f"\n训练指标结果：{result_path}")
    else:
        print(f"\n评估结果目录：{run_dir}")
    return return_code


def _python_available(executable: str) -> tuple[bool, str]:
    if os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
        path = Path(executable).expanduser()
        return path.exists(), str(path)
    found = shutil.which(executable)
    return bool(found), found or executable


def configured_dataset_path(settings: dict[str, Any]) -> Path:
    """Resolve the local dataset selected by the JSON configuration."""
    configured_path = get_nested(settings, "common.dataset_path")
    if configured_path:
        return project_path(str(configured_path)).resolve()

    dataset = str(get_nested(settings, "common.dataset", "sst2")).strip()
    candidate = Path(dataset).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return project_path(candidate).resolve()

    directory_name = "sst2" if dataset.casefold() in {"sst-2", "sst_2"} else dataset
    return (TOOL_ROOT / "datasets" / directory_name).resolve()


def doctor(
    method: Method,
    settings: dict[str, Any],
    *,
    check_dataset: bool = True,
    python_profile: str | None = None,
    display_title: str | None = None,
) -> int:
    failures = 0
    print(f"\n[{display_title or method.title}] 环境检查")
    print(f"代码目录：{method.root}")
    if method.root.exists():
        print("  [OK] 代码目录存在")
    else:
        print("  [FAIL] 代码目录不存在")
        failures += 1
    for path in method.required_files:
        if path.exists():
            print(f"  [OK] {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [FAIL] 缺少 {path}")
            failures += 1
    executable = configured_python(settings, python_profile or method.id)
    ok, resolved = _python_available(executable)
    print(f"  [{'OK' if ok else 'FAIL'}] Python: {resolved}")
    failures += 0 if ok else 1
    model = str(get_nested(settings, "common.model_7b", ""))
    if model.startswith("hf://") or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        print(f"  [INFO] 模型将从 Hugging Face 标识加载：{model}")
    elif model:
        model_path = project_path(model)
        exists = model_path.exists()
        print(f"  [{'OK' if exists else 'FAIL'}] 模型路径：{model_path}")
        failures += 0 if exists else 1
    if check_dataset:
        dataset_path = configured_dataset_path(settings)
        dataset_exists = dataset_path.exists()
        print(f"  [{'OK' if dataset_exists else 'FAIL'}] 数据集路径：{dataset_path}")
        failures += 0 if dataset_exists else 1
    for warning in method.warnings:
        print(f"  [WARN] {warning}")
    print(f"检查结论：{'通过' if failures == 0 else f'发现 {failures} 项阻塞'}")
    return 0 if failures == 0 else 1


def list_catalog(methods: dict[str, Method], selected_method: str | None = None) -> None:
    items = [methods[selected_method]] if selected_method else methods.values()
    for method in items:
        print(f"\n{method.id}: {method.title}")
        print(f"  {method.description}")
        for experiment in method.experiments:
            cost = "轻量" if not experiment.expensive else "GPU/耗时"
            print(f"  - {experiment.id:<22} [{cost}] {experiment.title}")
            print(f"    {experiment.description}")


def choose_from_menu(title: str, choices: list[tuple[str, str]]) -> str | None:
    print(f"\n{title}")
    for index, (_, label) in enumerate(choices, 1):
        print(f"  {index}. {label}")
    print("  0. 退出")
    while True:
        raw = input("请选择：").strip()
        if raw == "0":
            return None
        try:
            index = int(raw)
        except ValueError:
            print("请输入编号。")
            continue
        if 1 <= index <= len(choices):
            return choices[index - 1][0]
        print("编号超出范围。")
