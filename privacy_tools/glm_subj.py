"""ChatGLM3-6B/Subj experiment builders and manifest-backed history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import (
    PROJECT_ROOT,
    Experiment,
    Method,
    Step,
    get_nested,
    method_env,
    project_path,
    python_step,
)


GLM_ROOT = PROJECT_ROOT / "Glm-subj"
ATTACK_ROOT = GLM_ROOT / "privacy_attack"
ENTRY = GLM_ROOT / "attack_chatglm.py"
METHOD_NAMES = {"cliptext": "ClipText", "privsv": "PrivSV", "zo_dpga": "ZO-DPGA"}


def value(settings: dict[str, Any], key: str, default: Any = None) -> Any:
    return get_nested(settings, f"glm_subj.{key}", default)


def path(settings: dict[str, Any], key: str, default: str) -> Path:
    return project_path(str(value(settings, key, default))).resolve()


def _gpu_id(settings: dict[str, Any]) -> str:
    # method_env exposes configured physical GPU ids through CUDA_VISIBLE_DEVICES;
    # each Glm-subj process therefore addresses the first selected GPU as cuda:0.
    return "0"


def _step(settings: dict[str, Any], name: str, command: str, args: list[Any], run_dir: Path) -> Step:
    return python_step(
        settings, "glm_subj", name, ENTRY, [command, *args], run_dir,
        env=method_env(settings, GLM_ROOT),
    )


def finetune_steps(method_id: str, role: str, settings: dict[str, Any], run_dir: Path) -> list[Step]:
    """Create protected or matched non-protected Subj training/build commands."""
    base_model = str(value(settings, "base_model", get_nested(settings, "common.chatglm3_6b", "")))
    raw_data = path(settings, "raw_data_dir", "Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10")
    embedding_data = path(settings, "embedding_data_dir", "Glm-subj/data/subj_embedding_protected")
    models = run_dir / "models"
    gpu = _gpu_id(settings)

    if method_id == "cliptext":
        protected = role == "protected"
        return [_step(settings, "训练 ClipText+ Subj LoRA" if protected else "训练 Raw Subj LoRA 对照", "train-lora", [
            "--base-model", base_model,
            "--data-dir", embedding_data if protected else raw_data,
            "--output-dir", models / ("cliptext" if protected else "raw"),
            "--rank", value(settings, "lora_rank", 128),
            "--lora-alpha", value(settings, "lora_alpha", 256),
            "--epochs", value(settings, "lora_epochs", 15),
            "--batch-size", value(settings, "lora_batch_size", 1),
            "--learning-rate", value(settings, "lora_learning_rate", 1e-4),
            "--max-length", value(settings, "max_length", 512),
            "--gpu-id", gpu,
        ], run_dir)]

    if method_id == "zo_dpga":
        method = "dpzero" if role == "protected" else "mezo"
        return [_step(settings, f"训练 {method.upper()} causal LoRA", "train-zero-order", [
            "--method", method,
            "--base-model", base_model,
            "--data-dir", raw_data,
            "--output-dir", models / "gradient" / method,
            "--rank", value(settings, "zo_rank", 1),
            "--lora-alpha", value(settings, "zo_lora_alpha", 2),
            "--max-steps", value(settings, "zo_steps", 9000),
            "--batch-size", value(settings, "zo_batch_size", 64),
            "--learning-rate", value(settings, "zo_learning_rate", 0.005),
            "--zo-epsilon", value(settings, "zo_epsilon", 0.05),
            "--eval-steps", value(settings, "zo_eval_steps", 500),
            "--eval-samples", value(settings, "zo_eval_samples", 100),
            "--checkpoint-steps", value(settings, "zo_checkpoint_steps", 500),
            "--max-length", value(settings, "zo_max_length", 14),
            "--target-modules", value(settings, "zo_target_modules", "query_key_value"),
            "--layers-to-transform", value(settings, "zo_layers", 27),
            "--loss-prefix-tokens", value(settings, "zo_loss_prefix_tokens", 8),
            "--dp-epsilon", value(settings, "dp_epsilon", 8.0),
            "--dp-delta", value(settings, "dp_delta", 1e-5),
            "--clip-threshold", value(settings, "dpzero_clip", 0.1),
            "--gpu-id", gpu,
        ], run_dir)]

    # Example Steering consists of a raw causal LoRA plus a selected steering vector.
    variant = "private" if role == "protected" else "mean"
    raw_adapter = _latest_privsv_raw_adapter(settings)
    if raw_adapter is None:
        configured_raw = path(settings, "privsv_raw_adapter", "Glm-subj/outputs/embedding_full_parallel/models/raw")
        if (configured_raw / "adapter_config.json").is_file():
            raw_adapter = configured_raw
    train_raw = raw_adapter is None
    if train_raw:
        raw_adapter = models / "raw"
    steering_root = models / "example"
    steps: list[Step] = []
    if train_raw:
        steps.append(_step(settings, "训练 Raw Subj causal LoRA（PrivSV 共享底座）", "train-lora", [
            "--base-model", base_model,
            "--data-dir", raw_data,
            "--output-dir", raw_adapter,
            "--rank", value(settings, "lora_rank", 128),
            "--lora-alpha", value(settings, "lora_alpha", 256),
            "--epochs", value(settings, "lora_epochs", 15),
            "--batch-size", value(settings, "lora_batch_size", 1),
            "--learning-rate", value(settings, "lora_learning_rate", 1e-4),
            "--max-length", value(settings, "max_length", 512),
            "--gpu-id", gpu,
        ], run_dir))
    steps.append(_step(settings, f"构建 {'Private' if variant == 'private' else 'Mean'} Steering", "build-steering", [
            "--base-model", base_model,
            "--data-dir", raw_data,
            "--output-dir", steering_root,
            "--variants", variant,
            "--construction-samples", value(settings, "steering_samples", 7200),
            "--layers", *value(settings, "steering_layers", [11, 12, 13, 14, 15]),
            "--clip", value(settings, "steering_clip", 18.0),
            "--noise-multiplier", value(settings, "steering_noise_multiplier", 0.005),
            "--multiplier", value(settings, "steering_multiplier", 128.0),
            "--gpu-id", gpu,
        ], run_dir))
    return steps


def _attack_step(
    settings: dict[str, Any], run_dir: Path, role: str, *,
    adapter: Path, data_dir: Path, steering: Path | None = None,
) -> Step:
    base_model = str(value(settings, "base_model", get_nested(settings, "common.chatglm3_6b", "")))
    output = run_dir / "attacks" / role / "prefix"
    args: list[Any] = ["--base-model", base_model, "--adapter", adapter]
    if steering is not None:
        args += ["--steering", steering, "--steering-multiplier", value(settings, "steering_multiplier", 128.0)]
    args += [
        "--data-dir", data_dir,
        "--output-dir", output,
        "--targets-per-split", value(settings, "attack_targets", 100),
        "--samples-per-prefix", value(settings, "attack_samples_per_prefix", 1),
        "--prefix-tokens", value(settings, "attack_prefix_tokens", 2),
        "--min-suffix-tokens", value(settings, "attack_min_suffix_tokens", 8),
        "--max-new-tokens", value(settings, "attack_max_new_tokens", 128),
        "--temperature-start", value(settings, "attack_temperature_start", 1.0),
        "--temperature-end", value(settings, "attack_temperature_end", 1.0),
        "--top-p", value(settings, "attack_top_p", 1.0),
        "--top-k", value(settings, "attack_top_k", 50),
        "--seed", value(settings, "attack_seed", 42),
        "--gpu-id", _gpu_id(settings),
    ]
    return _step(settings, f"直接前缀攻击：{role}", "prefix-attack", args, run_dir)


def _latest_training_outputs(settings: dict[str, Any], method_id: str, experiment_id: str) -> list[Path]:
    for manifest, _run_dir in history_records(settings):
        if (manifest.get("method") == method_id and manifest.get("experiment") == experiment_id
                and manifest.get("model") == "chatglm3-6b" and manifest.get("dataset") == "subj"
                and manifest.get("status") == "completed" and _matches_current_training_context(manifest, settings)):
            outputs = manifest.get("artifacts", {}).get("model_outputs", [])
            existing = [Path(item) for item in outputs if Path(item).exists()]
            if existing:
                return existing
    return []


def _matches_current_training_context(manifest: dict[str, Any], settings: dict[str, Any]) -> bool:
    current_base = str(value(settings, "base_model", get_nested(settings, "common.chatglm3_6b", "")))
    current_raw = str(path(settings, "raw_data_dir", "Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10"))
    if manifest.get("base_model", current_base) != current_base or manifest.get("raw_data_dir", current_raw) != current_raw:
        return False
    training = manifest.get("training_config")
    if isinstance(training, dict):
        defaults = {
            "lora_rank": 128,
            "lora_alpha": 256,
            "lora_epochs": 15,
            "lora_batch_size": 1,
            "lora_learning_rate": 1e-4,
        }
        return all(training.get(key) == value(settings, key, default)
                   for key, default in defaults.items())
    return True


def _latest_privsv_raw_adapter(settings: dict[str, Any]) -> Path | None:
    for manifest, _run_dir in history_records(settings):
        if (manifest.get("method") == "privsv" and manifest.get("experiment") in {"utility", "baseline"}
                and manifest.get("model") == "chatglm3-6b" and manifest.get("dataset") == "subj"
                and manifest.get("status") == "completed" and _matches_current_training_context(manifest, settings)):
            candidates = [Path(item) for item in manifest.get("artifacts", {}).get("model_outputs", [])
                          if Path(item).name == "raw" and (Path(item) / "adapter_config.json").is_file()]
            if candidates:
                return candidates[0]
    return None


def direct_model_paths(settings: dict[str, Any], method_id: str) -> tuple[Path, Path, Path | None, Path | None]:
    """Return protected/baseline adapters and optional steering from latest runs or config."""
    if method_id == "cliptext":
        protected = path(settings, "cliptext_protected_adapter", "Glm-subj/outputs/embedding_full_parallel/models/embedding")
        baseline = path(settings, "cliptext_baseline_adapter", "Glm-subj/outputs/embedding_full_parallel/models/raw")
        if value(settings, "prefer_latest_training_runs", True):
            protected_runs = _latest_training_outputs(settings, method_id, "utility")
            baseline_runs = _latest_training_outputs(settings, method_id, "baseline")
            if protected_runs:
                protected = next((item for item in protected_runs if (item / "adapter_config.json").is_file()), protected)
            if baseline_runs:
                baseline = next((item for item in baseline_runs if (item / "adapter_config.json").is_file()), baseline)
        return protected, baseline, None, None
    if method_id == "zo_dpga":
        protected = path(settings, "zo_protected_adapter", "Glm-subj/outputs/gradient_natural_100_gpu23/models/gradient/dpzero")
        baseline = path(settings, "zo_baseline_adapter", "Glm-subj/outputs/gradient_natural_100_gpu23/models/gradient/mezo")
        if value(settings, "prefer_latest_training_runs", True):
            protected_runs = _latest_training_outputs(settings, method_id, "utility")
            baseline_runs = _latest_training_outputs(settings, method_id, "baseline")
            if protected_runs:
                protected = next((item for item in protected_runs if (item / "adapter_config.json").is_file()), protected)
            if baseline_runs:
                baseline = next((item for item in baseline_runs if (item / "adapter_config.json").is_file()), baseline)
        return protected, baseline, None, None

    raw_adapter = path(settings, "privsv_raw_adapter", "Glm-subj/outputs/embedding_full_parallel/models/raw")
    private = path(settings, "privsv_private_steering", "Glm-subj/outputs/example_reproduce/models/example/private")
    mean = path(settings, "privsv_mean_steering", "Glm-subj/outputs/example_reproduce/models/example/mean")
    if value(settings, "prefer_latest_training_runs", True):
        protected_runs = _latest_training_outputs(settings, method_id, "utility")
        baseline_runs = _latest_training_outputs(settings, method_id, "baseline")
        if protected_runs:
            raw_candidates = [item for item in protected_runs if item.name == "raw"]
            steering_candidates = [item for item in protected_runs if item.name == "example"]
            if steering_candidates:
                private = steering_candidates[0] / "private"
        if baseline_runs:
            steering_candidates = [item for item in baseline_runs if item.name == "example"]
            if steering_candidates:
                mean = steering_candidates[0] / "mean"
        raw_adapter = _latest_privsv_raw_adapter(settings) or raw_adapter
    return raw_adapter, raw_adapter, private, mean


def direct_attack_steps(method_id: str, settings: dict[str, Any], run_dir: Path) -> list[Step]:
    raw_data = path(settings, "raw_data_dir", "Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10")
    protected, baseline, protected_steering, baseline_steering = direct_model_paths(settings, method_id)
    labels = {"cliptext": ("ClipTextPlus", "RawLoRA"), "privsv": ("PrivateSteering", "MeanSteering"),
              "zo_dpga": ("DPZero", "MeZO")}[method_id]

    steps = [
        _attack_step(settings, run_dir, "protected", adapter=protected, data_dir=raw_data, steering=protected_steering),
        _attack_step(settings, run_dir, "baseline", adapter=baseline, data_dir=raw_data, steering=baseline_steering),
    ]
    compare = [
        "--require-matched-targets",
        "--result", METHOD_NAMES[method_id], "protected", run_dir / "attacks/protected/prefix/summary.json",
        "--result", labels[1], "baseline", run_dir / "attacks/baseline/prefix/summary.json",
        "--output-json", run_dir / "attacks/comparison.json",
        "--output-csv", run_dir / "attacks/comparison.csv",
    ]
    steps.append(_step(settings, "汇总保护方法与非保护对照攻击结果", "compare", compare, run_dir))
    return steps


def make_method(method_id: str, experiment_id: str, settings: dict[str, Any]) -> Method:
    name = METHOD_NAMES[method_id]
    if experiment_id in {"utility", "baseline"}:
        role = "protected" if experiment_id == "utility" else "baseline"
        builder = lambda current, run_dir: finetune_steps(method_id, role, current, run_dir)
        exp = Experiment(experiment_id, "保护方法" if role == "protected" else "效用对比方法",
                         "ChatGLM3-6B / Subj causal-LM 微调或配套非保护对照。", builder)
    elif experiment_id == "direct_attack":
        exp = Experiment(experiment_id, "直接评估", "Glm-subj known-prefix causal extraction 与配对比较。",
                         lambda current, run_dir: direct_attack_steps(method_id, current, run_dir))
    elif experiment_id == "indirect_attack":
        exp = Experiment(experiment_id, "间接评估占位", "ChatGLM3-6B / Subj 间接评估尚未接入。", lambda *_: [])
    else:
        raise ValueError(f"Unsupported Glm-subj experiment: {experiment_id}")
    return Method(method_id, name, f"{name} ChatGLM3-6B / Subj 后端", GLM_ROOT,
                  [ENTRY, ATTACK_ROOT / "train_lora.py", ATTACK_ROOT / "train_zo.py",
                   ATTACK_ROOT / "build_steering.py", ATTACK_ROOT / "prefix_attack.py",
                   ATTACK_ROOT / "compare.py"], [exp], path_name=f"glm-subj-{method_id}")


def output_root(settings: dict[str, Any]) -> Path:
    return path(settings, "output_dir", "Glm-subj/outputs")


def history_records(settings: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    root = output_root(settings)
    records: list[tuple[dict[str, Any], Path]] = []
    if not root.exists():
        return records
    for manifest_path in root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("backend") != "glm_subj":
            continue
        records.append((manifest, manifest_path.parent))
    return sorted(records, key=lambda item: str(item[0].get("created_at", "")), reverse=True)


def _legacy_history_files(settings: dict[str, Any], method_id: str, experiment_id: str) -> list[Path]:
    """Read the repository's pre-manifest Glm-subj outputs without mixing other backends."""
    root = output_root(settings)
    if experiment_id in {"utility", "baseline"}:
        if method_id == "cliptext":
            folder = "embedding" if experiment_id == "utility" else "raw"
            return [root / "embedding_full_parallel" / "models" / folder / "training_report.json"]
        if method_id == "zo_dpga":
            method = "dpzero" if experiment_id == "utility" else "mezo"
            return [root / "gradient_natural_100_gpu23" / "models" / "gradient" / method / "training_report.json"]
        return [root / "example_reproduce" / "models" / "example" / "build_report.json"]
    if experiment_id == "direct_attack":
        if method_id == "cliptext":
            base = root / "embedding_full_parallel" / "attacks_prefix2_random"
            return [base / "embedding" / "original" / "prefix" / "summary.json",
                    base / "raw" / "prefix" / "summary.json",
                    base / "embedding_original_comparison.json"]
        if method_id == "privsv":
            base = root / "example_reproduce" / "attacks" / "example"
            return [base / "private" / "prefix" / "summary.json",
                    base / "mean" / "prefix" / "summary.json",
                    base.parent / "example_direct_comparison.json"]
        base = root / "gradient_natural_100_gpu23"
        return [base / "attacks" / "gradient" / "dpzero" / "prefix" / "summary.json",
                base / "attacks" / "gradient" / "mezo" / "prefix" / "summary.json",
                base / "attacks" / "gradient_direct_comparison.json"]
    return []


def _display_history_file(resolved: Path, root: Path) -> None:
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    try:
        shown = resolved.relative_to(root)
    except ValueError:
        shown = resolved
    print(f"\n{shown}:")
    if resolved.name == "summary.json":
        for split in ("train", "holdout"):
            stats = data.get(split, {})
            print(f"  {split}: targets={stats.get('targets')} fuzzy={stats.get('fuzzy_success_rate')} exact={stats.get('exact_success_rate')}")
        print(f"  train_minus_holdout={data.get('train_minus_holdout')}")
    elif resolved.name == "comparison.json" or resolved.name.endswith("comparison.json"):
        for row in data.get("rows", []):
            print(f"  {row}")
        if data.get("pairwise_comparison"):
            print(f"  pairwise_comparison={data['pairwise_comparison']}")
    else:
        for key in ("method", "base_model", "data_dir", "elapsed_seconds", "privacy", "evaluation", "final_metrics"):
            if key in data:
                print(f"  {key}: {data[key]}")
        history = data.get("history")
        if isinstance(history, list) and history:
            latest = history[-1]
            print(f"  latest_epoch: {latest.get('epoch')}")
            print(f"  train: {latest.get('train', latest.get('online_train_loss'))}")
            print(f"  validation: {latest.get('validation')}")


def print_history(settings: dict[str, Any], method_id: str, experiment_id: str, *, role: str | None = None) -> None:
    expected_role = role or ("protected" if experiment_id == "utility" else "baseline" if experiment_id == "baseline" else None)
    matches = []
    for manifest, run_dir in history_records(settings):
        if (manifest.get("method") == method_id and manifest.get("experiment") == experiment_id
                and manifest.get("model") == "chatglm3-6b" and manifest.get("dataset") == "subj"
                and manifest.get("status") == "completed"
                and (expected_role is None or manifest.get("role") == expected_role)):
            matches.append((manifest, run_dir))
    title = METHOD_NAMES[method_id]
    print(f"\n===== Glm-subj {title} {experiment_id} 历史结果 =====")
    if not matches:
        legacy = [item for item in _legacy_history_files(settings, method_id, experiment_id) if item.is_file()]
        if not legacy:
            print("未找到匹配的已成功运行结果。")
            return
        print("读取 Glm-subj 中已有的历史产物（旧结果无统一 run manifest）：")
        for item in legacy:
            _display_history_file(item, output_root(settings))
        return
    manifest, run_dir = matches[0]
    artifacts = manifest.get("artifacts", {})
    candidates = []
    if experiment_id in {"utility", "baseline"}:
        candidates = [Path(item) for item in artifacts.get("reports", [])]
        if not candidates:
            candidates = list(run_dir.rglob("training_report.json")) + list(run_dir.rglob("build_report.json"))
    elif experiment_id == "direct_attack":
        candidates = [Path(item) for item in artifacts.get("summaries", [])]
        if not candidates:
            candidates = list((run_dir / "attacks").rglob("summary.json"))
        comparison = artifacts.get("comparison") or str(run_dir / "attacks/comparison.json")
        candidates.append(Path(comparison))
    printed = set()
    for report_path in candidates:
        resolved = report_path if report_path.is_absolute() else run_dir / report_path
        if resolved in printed or not resolved.is_file():
            continue
        printed.add(resolved)
        _display_history_file(resolved, output_root(settings))


def preflight(method_id: str, experiment_id: str, settings: dict[str, Any]) -> tuple[bool, str]:
    base_model = str(value(settings, "base_model", get_nested(settings, "common.chatglm3_6b", "")))
    if not base_model:
        return False, "未配置 ChatGLM3-6B 基座模型。"
    if Path(base_model).is_absolute() or base_model.startswith((".", "~")):
        if not project_path(base_model).exists():
            return False, f"ChatGLM3-6B 模型路径不存在：{project_path(base_model)}"
    if experiment_id in {"utility", "baseline", "direct_attack"}:
        raw = path(settings, "raw_data_dir", "Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10")
        if not raw.is_dir():
            return False, f"Subj 原始数据目录不存在：{raw}"
        if experiment_id == "utility" and method_id == "cliptext":
            embedding = path(settings, "embedding_data_dir", "Glm-subj/data/subj_embedding_protected")
            if not embedding.is_dir():
                return False, f"ClipText+ Subj 数据目录不存在：{embedding}"
    if experiment_id == "direct_attack":
        protected, baseline, private_steering, mean_steering = direct_model_paths(settings, method_id)
        needed = [protected, baseline]
        if private_steering is not None:
            needed += [private_steering, mean_steering]
        missing = [item for item in needed if not item.exists()]
        if missing:
            return False, "直接评估缺少已完成的模型/steering 产物：" + ", ".join(map(str, missing))
    return True, ""


def manifest_metadata(method_id: str, experiment_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    return {
        "backend": "glm_subj", "model": "chatglm3-6b", "dataset": "subj",
        "method": method_id, "experiment": experiment_id,
        "role": "protected" if experiment_id in {"utility", "direct_attack"} else "baseline" if experiment_id == "baseline" else "placeholder",
        "base_model": str(value(settings, "base_model", get_nested(settings, "common.chatglm3_6b", ""))),
        "raw_data_dir": str(path(settings, "raw_data_dir", "Glm-subj/data/subj_raw_prepared/subj/none_a3c561096c597e10")),
        "training_config": {
            key: value(settings, key, default) for key, default in (
                ("lora_rank", 128), ("lora_alpha", 256), ("lora_epochs", 15),
                ("lora_batch_size", 1), ("lora_learning_rate", 1e-4),
            )
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
