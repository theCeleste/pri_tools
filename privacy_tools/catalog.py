from __future__ import annotations

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


CLIP_ROOT = PROJECT_ROOT / "ClipText"
CLIP_SST = CLIP_ROOT / "sst" / "ClipText" / "ClipText" / "sst2"
CLIP_ATTACK = CLIP_ROOT / "clip-attack"
INTEGRATION_ROOT = PROJECT_ROOT / "privacy_tools"
DIRECT_ATTACK_RUNNER = INTEGRATION_ROOT / "direct_attack_runner.py"
PROMPT_DIRECT_ATTACK_RUNNER = INTEGRATION_ROOT / "prompt_direct_attack_runner.py"
INDIRECT_ATTACK_RUNNER = INTEGRATION_ROOT / "indirect_attack_runner.py"
NATIVE_CSV_ATTACK_EVALUATOR = INTEGRATION_ROOT / "native_csv_attack_evaluator.py"
ATTACK_EVALUATOR = INTEGRATION_ROOT / "attack_evaluator.py"
NEIGHBOUR_ATTACK_EVALUATOR = INTEGRATION_ROOT / "neighbour_attack_evaluator.py"
CLIP_ISOLATED_RUNNER = INTEGRATION_ROOT / "cliptext_isolated_runner.py"
REFERENCE_ATTACK_ROOT = INTEGRATION_ROOT / "reference_assets" / "shared_legacy_attack"
SHARED_SST2_ROOT = INTEGRATION_ROOT / "datasets" / "sst2"
PTCDS_ROOT = PROJECT_ROOT / "PTCDS"
PTCDS_ATTACK = PTCDS_ROOT / "attack"
KAL_ROOT = PROJECT_ROOT / "Kal-DP"
KAL_ATTACK = KAL_ROOT / "attack"
ZO_ROOT = PROJECT_ROOT / "ZO-DPGA"
ZO_REPRODUCTION = ZO_ROOT / "run_zo_dpga_858_reproduction.py"
DADP_ROOT = PROJECT_ROOT / "DADP-ICL"
PRIVSV_ROOT = PROJECT_ROOT / "PrivSV"


# User-facing task topology. Command builders remain registered on Method below.
FINETUNE_METHOD_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "gradient": ("梯度扰动", ("kal_dp", "zo_dpga")),
    "embedding": ("嵌入扰动", ("cliptext", "ptcds")),
    "example": ("示例扰动", ("dadp_icl", "privsv")),
}

EVALUATION_METHOD_GROUPS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "explicit": {
        "gradient": ("梯度扰动", ("kal_dp", "zo_dpga")),
        "embedding": ("嵌入扰动", ("cliptext", "ptcds")),
        "none": ("无扰动", ("none",)),
    },
    "implicit": {
        "example": ("示例扰动", ("dadp_icl", "privsv")),
        "none": ("无扰动", ("none",)),
    },
}


def _method_value(settings: dict[str, Any], method: str, key: str, default: Any) -> Any:
    return get_nested(settings, f"methods.{method}.{key}", default)


def _common_value(settings: dict[str, Any], key: str, default: Any) -> Any:
    return get_nested(settings, f"common.{key}", default)


def _as_path(value: Any) -> Path:
    return project_path(str(value)).resolve()


def _source_string(value: Any) -> str:
    """转义将被嵌入 Python 普通字符串字面量的 Windows 路径。"""
    return str(value).replace("\\", "\\\\")


def _present_replacements(source: Path, candidates: dict[str, str]) -> dict[str, str]:
    """只保留源文件里实际存在的硬编码片段。"""
    text = source.read_text(encoding="utf-8")
    return {old: new for old, new in candidates.items() if old in text}


def _clip_common_args(settings: dict[str, Any]) -> list[Any]:
    save_stop_words = _method_value(settings, "cliptext", "save_stop_words", True)
    args: list[Any] = [
        "--dataset", "sst2",
        "--eps", _method_value(settings, "cliptext", "epsilon", 10.0),
        "--top_k", _method_value(settings, "cliptext", "top_k", 20),
        "--embedding_type", "glove_840B-300d",
        "--mapping_strategy", "conservative",
        "--privatization_strategy", "s1",
    ]
    # 原脚本使用 argparse(type=bool)：传入字符串 "False" 仍会被解析为 True。
    # False 时不传该参数（原脚本默认值即 False），True 时才显式传入。
    if bool(save_stop_words):
        args += ["--save_stop_words", "True"]
    return args


def clip_sanitize(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    epsilon = _method_value(settings, "cliptext", "epsilon", 10.0)
    top_k = _method_value(settings, "cliptext", "top_k", 20)
    return [python_step(
        settings,
        "cliptext",
        "构造候选词并按指数机制生成脱敏 SST-2",
        CLIP_ISOLATED_RUNNER,
        [
            "--source-root", CLIP_SST,
            "--entry", CLIP_SST / "main.py",
            "--workspace", run_dir / "cliptext_workspace",
            "--eps", epsilon,
            "--top-k", top_k,
            "--",
            *_clip_common_args(settings),
        ],
        run_dir,
        env=method_env(settings, CLIP_SST),
    )]


def clip_bert_utility(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    epsilon = _method_value(settings, "cliptext", "epsilon", 10.0)
    top_k = _method_value(settings, "cliptext", "top_k", 20)
    args = _clip_common_args(settings) + [
        "--batch_size", _method_value(settings, "cliptext", "batch_size", 32),
        "--epochs", _method_value(settings, "cliptext", "epochs", 3),
        "--model_type", _method_value(settings, "cliptext", "bert_model", "bert-base-uncased"),
    ]
    return [python_step(
        settings,
        "cliptext",
        "ClipText 脱敏并进行 BERT 效用评估",
        CLIP_ISOLATED_RUNNER,
        [
            "--source-root", CLIP_SST,
            "--entry", CLIP_ROOT / "sst" / "clip" / "main.py",
            "--workspace", run_dir / "cliptext_workspace",
            "--eps", epsilon,
            "--top-k", top_k,
            "--",
            *args,
        ],
        run_dir,
        env=method_env(settings, CLIP_SST),
    )]


def clip_llama_utility(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    epsilon = _method_value(settings, "cliptext", "epsilon", 10.0)
    top_k = _method_value(settings, "cliptext", "top_k", 20)
    save = bool(_method_value(settings, "cliptext", "save_stop_words", True))
    default_train = CLIP_SST / "privatized_dataset" / "glove_840B-300d" / "conservative" / f"eps_{epsilon}_top_{top_k}_s1_save_stop_words_{save}"
    train_path = _as_path(_method_value(settings, "cliptext", "train_data_path", default_train))
    val_path = _as_path(_method_value(settings, "cliptext", "validation_data_path", CLIP_SST / "datasets" / "sst2" / "dev.tsv"))
    return [python_step(
        settings,
        "cliptext",
        "用 LoRA 在脱敏文本上训练 Llama 分类器",
        CLIP_ROOT / "dp" / "cliptext.py",
        [
            "--train_data_path", train_path,
            "--val_data_path", val_path,
            "--batch_size", _method_value(settings, "cliptext", "batch_size", 32),
            "--epochs", _method_value(settings, "cliptext", "epochs", 3),
            "--lora_rank", _method_value(settings, "cliptext", "lora_rank", 16),
            "--learning_rate", _method_value(settings, "cliptext", "learning_rate", 1e-4),
            "--gpu_id", 0,
            "--model_name", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        ],
        run_dir,
        env=method_env(settings, CLIP_ROOT / "dp"),
    )]


def clip_baseline(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    """Run the same LoRA classifier on the original, non-obfuscated SST-2 data."""
    return [python_step(
        settings,
        "cliptext",
        "在原始 SST-2 文本上训练 LoRA 对比模型",
        CLIP_ROOT / "dp" / "cliptext.py",
        [
            "--train_data_path", SHARED_SST2_ROOT / "train.tsv",
            "--val_data_path", SHARED_SST2_ROOT / "dev.tsv",
            "--batch_size", _method_value(settings, "cliptext", "batch_size", 32),
            "--epochs", _method_value(settings, "cliptext", "epochs", 3),
            "--lora_rank", _method_value(settings, "cliptext", "lora_rank", 16),
            "--learning_rate", _method_value(settings, "cliptext", "learning_rate", 1e-4),
            "--gpu_id", 0,
            "--model_name", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        ],
        run_dir,
        env=method_env(settings, CLIP_ROOT / "dp"),
    )]


def _readme_prefix_attack(
    settings: dict[str, Any],
    run_dir: Path,
    method_id: str,
    attack_root: Path,
    protected_adapter_literal: str,
    baseline_adapter_literal: str,
) -> list[Step]:
    """Execute the protected/baseline extraction sequence documented by the original READMEs."""
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    protected_adapter = _as_path(_method_value(settings, method_id, "adapter_path", ""))
    baseline_adapter = _as_path(_method_value(settings, method_id, "baseline_adapter_path", ""))
    attack_samples = max(1, int(_method_value(settings, method_id, "attack_samples", 20)))
    protected_source = attack_root / "extraction_temperature_decay.py"
    baseline_source = attack_root / "extraction_temperature_decay_wo.py"
    member_source = attack_root / "member_test.py"

    def attack_replacements(source: Path, adapter_literal: str, adapter: Path) -> dict[str, str]:
        return _present_replacements(source, {
            "../../models/Llama-2-7b-hf": _source_string(model),
            adapter_literal: _source_string(adapter),
        })

    def member_replacements(output: Path) -> dict[str, str]:
        return _present_replacements(member_source, {
            'tsv_path = "../../privacy_tools/datasets/sst2/train.tsv"': (
                f'tsv_path = r"{_source_string(SHARED_SST2_ROOT / "train.tsv")}"'
            ),
            'attack_path = "./results/extraction_temperature_decay.txt"': (
                f'attack_path = r"{_source_string(output)}"'
            ),
            'attack_path = "./results/extraction_temperature_decay_wo.txt"': (
                f'attack_path = r"{_source_string(output)}"'
            ),
        })

    steps: list[Step] = []
    for label, source, adapter_literal, adapter in (
        ("protected", protected_source, protected_adapter_literal, protected_adapter),
        ("baseline", baseline_source, baseline_adapter_literal, baseline_adapter),
    ):
        output = run_dir / f"{label}_extraction_temperature_decay.txt"
        steps.extend([
            python_step(
                settings,
                method_id,
                f"按原始说明执行{('受保护' if label == 'protected' else '未保护')}模型前缀抽取",
                source,
                [
                    "--N", attack_samples,
                    "--batch_size", min(6, attack_samples),
                    "--outfile", output,
                ],
                run_dir,
                env=method_env(settings, attack_root),
                replacements=attack_replacements(source, adapter_literal, adapter),
            ),
            python_step(
                settings,
                method_id,
                f"按原始 member_test 统计{('受保护' if label == 'protected' else '未保护')}攻击命中率",
                member_source,
                [],
                run_dir,
                env=method_env(settings, attack_root),
                replacements=member_replacements(output),
            ),
        ])
    return steps


def _prefix_existing(settings: dict[str, Any], run_dir: Path, method_id: str, attack_root: Path) -> list[Step]:
    train_tsv = SHARED_SST2_ROOT / "train.tsv"
    existing = attack_root / "extraction_temperature_decay.txt"
    return [python_step(
        settings,
        method_id,
        "复核共享遗留前缀攻击样例（不作为方法独立结论）",
        ATTACK_EVALUATOR,
        [
            "--samples", existing,
            "--train-tsv", train_tsv,
            "--output", run_dir / "shared_legacy_attack_evaluation.json",
            "--legacy-format",
            "--scope", "shared_legacy_reference",
        ],
        run_dir,
        env=method_env(settings, attack_root),
    )]


def _neighbour_existing(settings: dict[str, Any], run_dir: Path, method_id: str, data_root: Path) -> list[Step]:
    return [python_step(
        settings,
        method_id,
        "用统一阈值复核已有邻居成员推断结果",
        NEIGHBOUR_ATTACK_EVALUATOR,
        [
            "--base-train-pkl", data_root / "wo_all_scores_sst2_roberta_0_train.pkl",
            "--base-test-pkl", data_root / "wo_all_scores_sst2_roberta_0_test.pkl",
            "--priv-train-pkl", data_root / "all_scores_sst2_roberta_0_train.pkl",
            "--priv-test-pkl", data_root / "all_scores_sst2_roberta_0_test.pkl",
            "--target-fprs", 0.01, 0.001, 0.0001,
            "--pos-neg-ratio", 1.0,
            "--seed", 0,
            "--output", run_dir / "shared_legacy_neighbour_mia.csv",
        ],
        run_dir,
        env=method_env(settings),
    )]


def clip_prefix_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _readme_prefix_attack(
        settings, run_dir, "cliptext", CLIP_ATTACK,
        "../../Kal-DP/kal_dp_adapter_best", "../../Kal-DP/lora_wo_sst2_model_best",
    )


def clip_prefix_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _prefix_existing(settings, run_dir, "cliptext", REFERENCE_ATTACK_ROOT)


def clip_neighbour_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _neighbour_existing(settings, run_dir, "cliptext", REFERENCE_ATTACK_ROOT)


def _ptcds_lora_train(settings: dict[str, Any], run_dir: Path, baseline: bool) -> list[Step]:
    """Run sections 2/3 of PTCDS/md.md with portable paths and output location."""
    source = PTCDS_ROOT / "finetune_llama2_sst2_lora_data.py"
    protected_train = PTCDS_ROOT / "output_PTCDS_plus_glove" / "SST-2" / "eps_6.00" / "sword_0.90_p_0.30" / "train.tsv"
    train_path = (
        PTCDS_ROOT / "data" / "SST-2" / "train.tsv"
        if baseline
        else _as_path(_method_value(settings, "ptcds", "train_data_path", protected_train))
    )
    return [python_step(
        settings,
        "ptcds",
        "基于原始数据进行 base LoRA 微调" if baseline else "基于 PTCDS 嵌入扰动方法进行 LoRA 微调",
        source,
        [
            "--model_name_or_path", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
            "--train_path", train_path,
            "--output_dir", run_dir / "model_output",
            "--eval_steps", 948,
            "--wandb_project", "llama2-sst2-lora",
            "--run_name", "llama2-7b-sst2",
        ],
        run_dir,
        env=method_env(settings, PTCDS_ROOT),
    )]


def ptcds_train(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _ptcds_lora_train(settings, run_dir, False)


def ptcds_baseline(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _ptcds_lora_train(settings, run_dir, True)


def ptcds_prefix_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    """Use the ClipText two-branch extraction flow with PTCDS weights and outputs."""
    protected_source = PTCDS_ATTACK / "extraction_temperature_decay.py"
    baseline_source = PTCDS_ATTACK / "extraction_temperature_decay_base.py"
    member_source = PTCDS_ATTACK / "member_test.py"
    attack_samples = max(1, int(_method_value(settings, "ptcds", "attack_samples", 20)))
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    protected_adapter = _as_path(_method_value(
        settings,
        "ptcds",
        "adapter_path",
        PTCDS_ROOT / "outputs" / "llama2-sst2-lora-esp6-4.20" / "best_model",
    ))
    baseline_adapter = _as_path(_method_value(
        settings,
        "ptcds",
        "baseline_adapter_path",
        PTCDS_ROOT / "outputs" / "llama2-sst2-lora-base-4.20" / "best_model",
    ))

    def member_replacements(output: Path) -> dict[str, str]:
        return _present_replacements(member_source, {
            'tsv_path = "../../privacy_tools/datasets/sst2/train.tsv"': (
                f'tsv_path = r"{_source_string(SHARED_SST2_ROOT / "train.tsv")}"'
            ),
            'attack_path = "./results/extraction_temperature_decay_base.txt"': (
                f'attack_path = r"{_source_string(output)}"'
            ),
        })

    protected_output = run_dir / "protected_extraction_temperature_decay.txt"
    baseline_output = run_dir / "baseline_extraction_temperature_decay.txt"
    return [
        python_step(
            settings,
            "ptcds",
            "执行 PTCDS 受保护模型前缀抽取",
            protected_source,
            [
                "--N", attack_samples,
                "--batch_size", min(6, attack_samples),
                "--outfile", protected_output,
                "--model_name", model,
                "--baseline_adapter", protected_adapter,
                "--protected_adapter", protected_adapter,
            ],
            run_dir,
            env=method_env(settings, PTCDS_ATTACK),
        ),
        python_step(
            settings,
            "ptcds",
            "统计 PTCDS 受保护模型攻击命中率",
            member_source,
            [],
            run_dir,
            env=method_env(settings, PTCDS_ATTACK),
            replacements=member_replacements(protected_output),
        ),
        python_step(
            settings,
            "ptcds",
            "执行 PTCDS 未保护 base 模型前缀抽取",
            baseline_source,
            [
                "--N", attack_samples,
                "--batch_size", min(6, attack_samples),
                "--outfile", baseline_output,
            ],
            run_dir,
            env=method_env(settings, PTCDS_ATTACK),
            replacements=_present_replacements(baseline_source, {
                "../../models/Llama-2-7b-hf": _source_string(model),
                "../outputs/llama2-sst2-lora-base/base-checkpoint-6500": _source_string(baseline_adapter),
            }),
        ),
        python_step(
            settings,
            "ptcds",
            "统计 PTCDS 未保护 base 模型攻击命中率",
            member_source,
            [],
            run_dir,
            env=method_env(settings, PTCDS_ATTACK),
            replacements=member_replacements(baseline_output),
        ),
    ]


def ptcds_prefix_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return [python_step(
        settings,
        "ptcds",
        "复核 PTCDS 专属历史前缀抽取结果",
        ATTACK_EVALUATOR,
        [
            "--samples", PTCDS_ATTACK / "results" / "extraction_temperature_decay_eps3.txt",
            "--train-tsv", SHARED_SST2_ROOT / "train.tsv",
            "--output", run_dir / "ptcds_existing_direct_attack.json",
            "--legacy-format",
            "--scope", "ptcds_method_specific_reference",
        ],
        run_dir,
        env=method_env(settings, PTCDS_ATTACK),
    )]


def ptcds_neighbour_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return [python_step(
        settings,
        "ptcds",
        "复核 PTCDS 专属邻居成员推断结果",
        NEIGHBOUR_ATTACK_EVALUATOR,
        [
            "--base-train-pkl", PTCDS_ATTACK / "wo_all_scores_sst2_roberta_0_train.pkl",
            "--base-test-pkl", PTCDS_ATTACK / "wo_all_scores_sst2_roberta_0_test.pkl",
            "--priv-train-pkl", PTCDS_ATTACK / "all_scores_sst2_roberta_0_train.pkl",
            "--priv-test-pkl", PTCDS_ATTACK / "all_scores_sst2_roberta_0_test.pkl",
            "--target-fprs", 0.01, 0.001, 0.0001,
            "--pos-neg-ratio", 1.0,
            "--seed", 0,
            "--output", run_dir / "ptcds_neighbour_mia.csv",
            "--scope", "ptcds_method_specific_reference",
            "--method-specific",
        ],
        run_dir,
        env=method_env(settings, PTCDS_ATTACK),
    )]


def _patch_dp_accounting(source: Path, epsilon: float, delta: float, epochs: int) -> dict[str, str]:
    text = source.read_text(encoding="utf-8")
    replacements: dict[str, str] = {}
    # 两份原脚本把隐私预算写在 make_private_with_epsilon 调用中；只改运行时副本。
    for old, new in [
        ("target_epsilon=8,", f"target_epsilon={epsilon},"),
        ("target_delta=5e-6,", f"target_delta={delta},"),
        ("epochs=2,", f"epochs={epochs},"),
    ]:
        if old in text:
            replacements[old] = new
    return replacements


def _kal_train(settings: dict[str, Any], run_dir: Path, baseline: bool) -> list[Step]:
    source = KAL_ROOT / ("dpsgd.py" if baseline else "kal_dp.py")
    epsilon = float(_method_value(settings, "kal_dp", "epsilon", 8.0))
    delta = float(_method_value(settings, "kal_dp", "delta", 5e-6))
    epochs = int(_method_value(settings, "kal_dp", "epochs", 2))
    return [python_step(
        settings,
        "kal_dp",
        "DP-SGD 对照训练" if baseline else "Kalman 梯度过滤 + DP 训练",
        source,
        [
            "--batch_size", _method_value(settings, "kal_dp", "batch_size", 32),
            "--max_physical_batch_size", _method_value(
                settings,
                "kal_dp",
                "dpsgd_max_physical_batch_size" if baseline else "max_physical_batch_size",
                12 if baseline else 16,
            ),
            "--seq_length", 128,
            "--is_lora", "True",
            "--lora_rank", _method_value(settings, "kal_dp", "lora_rank", 16),
            "--learning_rate", _method_value(settings, "kal_dp", "learning_rate", 1e-4),
            "--max_grad_norm", _method_value(settings, "kal_dp", "max_grad_norm", 3.0),
            "--epochs", epochs,
            "--model_name", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        ],
        run_dir,
        env=method_env(settings, KAL_ROOT),
        replacements=_patch_dp_accounting(source, epsilon, delta, epochs),
    )]


def kal_train(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _kal_train(settings, run_dir, False)


def dpsgd_train(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _kal_train(settings, run_dir, True)


def peft_train(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return [python_step(
        settings,
        "kal_dp",
        "非隐私 LoRA/PEFT 对照训练",
        KAL_ROOT / "Peft.py",
        [
            "--train_data_path", SHARED_SST2_ROOT / "train.tsv",
            "--val_data_path", SHARED_SST2_ROOT / "dev.tsv",
            "--batch_size", _method_value(settings, "kal_dp", "batch_size", 32),
            "--epochs", _method_value(settings, "kal_dp", "epochs", 2),
            "--gpu_id", 0,
            "--model_name", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        ],
        run_dir,
        env=method_env(settings, KAL_ROOT),
    )]


def kal_prefix_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _readme_prefix_attack(
        settings, run_dir, "kal_dp", KAL_ATTACK,
        "../lora_kal_dp_sst2_model_best", "../lora_wo_sst2_model_best",
    )


def kal_prefix_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _prefix_existing(settings, run_dir, "kal_dp", REFERENCE_ATTACK_ROOT)


def kal_neighbour_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _neighbour_existing(settings, run_dir, "kal_dp", REFERENCE_ATTACK_ROOT)


def zo_reproduce(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    source = ZO_REPRODUCTION
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    return [python_step(
        settings,
        "zo_dpga",
        "复现冻结的 ZO-DPGA 8.58 配置",
        source,
        ["--output-dir", run_dir / "model_output", "--run"],
        run_dir,
        env=method_env(settings, ZO_ROOT),
        replacements={
            '"model_name": config["model_name"],': f'"model_name": "{_source_string(model)}",',
        },
    )]


def zo_baseline(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    source = ZO_ROOT / "run_lora_mezo_892_reproduction.py"
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    return [python_step(
        settings,
        "zo_dpga",
        "运行 LoRA + MeZO 对比微调",
        source,
        ["--output-dir", run_dir / "model_output", "--run"],
        run_dir,
        env=method_env(settings, ZO_ROOT),
        replacements={
            '"model_name": config["model_name"],': f'"model_name": "{_source_string(model)}",',
        },
    )]


def zo_formal(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    source = ZO_ROOT / "run_formal_experiment.py"
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    return [python_step(
        settings,
        "zo_dpga",
        "执行正式 ZO-DPGA 隐私预算实验中的一个任务",
        source,
        [
            "--suite", _method_value(settings, "zo_dpga", "formal_suite", "zo_dpga_budget_sweep"),
            "--index", _method_value(settings, "zo_dpga", "formal_index", 0),
            "--run",
        ],
        ZO_ROOT,
        env=method_env(settings, ZO_ROOT),
        replacements={
            "LLAMA_DIR = Path(__file__).resolve().parent": f'LLAMA_DIR = Path(r"{_source_string(ZO_ROOT)}")',
            '"model_name": config["model_name"],': f'"model_name": "{_source_string(model)}",',
            'output_dir = Path(config["output_root"]) / suite_name / experiment_id': f'output_dir = Path(r"{_source_string(run_dir / "model_output")}")',
        },
    )]


def zo_validate(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    names = ["validate_zo_math.py", "validate_dp_privacy.py", "validate_858_reproduction.py"]
    return [
        python_step(settings, "zo_dpga", f"校验 {name}", ZO_ROOT / "tests" / name, [], run_dir, env=method_env(settings, ZO_ROOT))
        for name in names
    ]


def zo_summarize(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return [python_step(
        settings,
        "zo_dpga",
        "汇总仓库已有 benchmark/eval 指标",
        ZO_ROOT / "summarize_benchmarks.py",
        ["--root", ZO_ROOT / "result", "--output", run_dir / "benchmark_summary"],
        run_dir,
        env=method_env(settings, ZO_ROOT),
    )]


def zo_prefix_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    # ZO-DPGA's README explicitly specifies the Kal-DP attack procedure with model paths changed.
    return _readme_prefix_attack(
        settings, run_dir, "zo_dpga", KAL_ATTACK,
        "../lora_kal_dp_sst2_model_best", "../lora_wo_sst2_model_best",
    )


def _fresh_indirect_attack(
    settings: dict[str, Any],
    run_dir: Path,
    method_id: str,
    method_root: Path,
    protection: str,
) -> list[Step]:
    """Regenerate neighbour scores against this method's current protected artefact."""
    args: list[Any] = [
        "--method", method_id,
        "--model", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        "--mask-model", _common_value(settings, "mask_model", "roberta-base"),
        "--nonmember-tsv", SHARED_SST2_ROOT / "test.tsv",
        "--samples", max(1, int(_method_value(settings, method_id, "indirect_samples", 20))),
        "--neighbors", max(1, int(_method_value(settings, method_id, "indirect_neighbors", 3))),
        "--max-length", max(32, int(_method_value(settings, method_id, "indirect_max_length", 256))),
        "--seed", int(_method_value(settings, method_id, "indirect_seed", 0)),
        "--output-dir", run_dir / "fresh_indirect_attack",
    ]
    if protection == "adapter":
        args += [
            "--member-tsv", SHARED_SST2_ROOT / "train.tsv",
            "--adapter", _as_path(_method_value(settings, method_id, "adapter_path", "")),
        ]
    elif protection == "prompt":
        prompt_data = _as_path(_method_value(
            settings, method_id, "prompt_data_path", DADP_ROOT / "synthetic_text_results.json"
        ))
        args += [
            "--member-json", prompt_data,
            "--prompt-data", prompt_data,
            "--prompt-examples", max(1, int(_method_value(settings, method_id, "prompt_examples", 5))),
        ]
    elif protection == "steering":
        checkpoint = _as_path(_method_value(
            settings,
            method_id,
            "steering_checkpoint_path",
            PRIVSV_ROOT / "private_steering_vectors_llama_7b_hf.pt",
        ))
        args += [
            "--member-json", PRIVSV_ROOT / "datasets" / "generate" / "sst2" / "generate_dataset.json",
            "--steering-checkpoint", checkpoint,
            "--steering-wrapper", PRIVSV_ROOT / "steering_wrapper.py",
            "--steering-multiplier", _method_value(settings, method_id, "steering_multiplier", 2.0),
        ]
    else:
        raise ValueError(f"未知间接攻击保护模式：{protection}")
    return [python_step(
        settings,
        method_id,
        "重新生成邻居分数并执行成员推断攻击",
        INDIRECT_ATTACK_RUNNER,
        args,
        run_dir,
        env=method_env(settings, method_root),
    )]


def _native_adapter_indirect_attack(
    settings: dict[str, Any],
    run_dir: Path,
    method_id: str,
    attack_root: Path,
    protected_adapter_literal: str,
    baseline_adapter_literal: str,
) -> list[Step]:
    """Run the original protected/baseline neighbour-MIA generators from a runtime copy."""
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    mask_model = str(_common_value(settings, "mask_model", "roberta-base"))
    protected_adapter = _as_path(_method_value(settings, method_id, "adapter_path", ""))
    baseline_adapter = _as_path(_method_value(settings, method_id, "baseline_adapter_path", ""))
    sample_count = max(1, int(_method_value(settings, method_id, "native_attack_samples", 20)))
    visible_gpus = [item.strip() for item in str(_common_value(settings, "gpu", "0")).split(",") if item.strip()]
    search_gpu = 1 if len(visible_gpus) > 1 else 0
    protected_source = attack_root / "neighbour_attack.py"
    baseline_source = attack_root / "neighbour_attack_wo.py"

    common_candidates = {
        "../../models/Llama-2-7b-hf": _source_string(model),
        "../../privacy_tools/datasets/sst2/train.tsv": _source_string(SHARED_SST2_ROOT / "train.tsv"),
        "../../privacy_tools/datasets/sst2/test.tsv": _source_string(SHARED_SST2_ROOT / "test.tsv"),
        "../../privacy_tools/datasets/sst2/dev.tsv": _source_string(SHARED_SST2_ROOT / "dev.tsv"),
        'RobertaTokenizer.from_pretrained("./models/roberta-base")': f'RobertaTokenizer.from_pretrained("{_source_string(mask_model)}")',
        'RobertaForMaskedLM.from_pretrained("./models/roberta-base")': f'RobertaForMaskedLM.from_pretrained("{_source_string(mask_model)}")',
        'RobertaTokenizer.from_pretrained("../../models/roberta-base")': f'RobertaTokenizer.from_pretrained("{_source_string(mask_model)}")',
        'RobertaForMaskedLM.from_pretrained("../../models/roberta-base")': f'RobertaForMaskedLM.from_pretrained("{_source_string(mask_model)}")',
        "    batch_size = 500": f"    batch_size = {sample_count}",
    }

    def replacements(source: Path, adapter_literal: str, adapter: Path) -> dict[str, str]:
        candidates = dict(common_candidates)
        candidates[adapter_literal] = _source_string(adapter)
        return _present_replacements(source, candidates)

    steps: list[Step] = []
    for protected, source, adapter_literal, adapter, prefix in (
        (True, protected_source, protected_adapter_literal, protected_adapter, "受保护"),
        (False, baseline_source, baseline_adapter_literal, baseline_adapter, "非保护基线"),
    ):
        for split in ("train", "test"):
            steps.append(python_step(
                settings,
                method_id,
                f"原生邻居攻击：{prefix}模型 / {split}",
                source,
                [
                    "--proc-id", 0,
                    "--model", "roberta",
                    "--dataset", "sst2",
                    "--split", split,
                    "--attack-gpu", 0,
                    "--search-gpu", search_gpu,
                ],
                run_dir,
                env=method_env(settings, attack_root),
                replacements=replacements(source, adapter_literal, adapter),
            ))
    steps.append(python_step(
        settings,
        method_id,
        "按原始 README 参数比较受保护与未保护邻居成员推断结果",
        attack_root / "evaluate_mia_fair_compare.py",
        [
            "--base-train-pkl", run_dir / "wo_all_scores_sst2_roberta_0_train.pkl",
            "--base-test-pkl", run_dir / "wo_all_scores_sst2_roberta_0_test.pkl",
            "--priv-train-pkl", run_dir / "all_scores_sst2_roberta_0_train.pkl",
            "--priv-test-pkl", run_dir / "all_scores_sst2_roberta_0_test.pkl",
            "--target-fprs", 0.025, 0.01,
            "--pos-neg-ratio", 0.2,
            "--seed", 0,
        ],
        run_dir,
        env=method_env(settings, attack_root),
    ))
    return steps


def clip_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    # The checked-in ClipText scripts accidentally point at Kal-DP. Runtime-only replacement
    # binds them to the ClipText protected and baseline adapters supplied by the user.
    return _native_adapter_indirect_attack(
        settings, run_dir, "cliptext", CLIP_ATTACK,
        "../../Kal-DP/kal_dp_adapter_best", "../../Kal-DP/lora_wo_sst2_model_best",
    )


def ptcds_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _native_adapter_indirect_attack(
        settings, run_dir, "ptcds", PTCDS_ATTACK,
        "../outputs/llama2-sst2-lora-base/eps3-checkpoint-7500",
        "../outputs/llama2-sst2-lora-base/base-checkpoint-6500",
    )


def kal_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _native_adapter_indirect_attack(
        settings, run_dir, "kal_dp", KAL_ATTACK,
        "../lora_kal_dp_sst2_model_best", "../lora_wo_sst2_model_best",
    )


def zo_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _native_adapter_indirect_attack(
        settings, run_dir, "zo_dpga", KAL_ATTACK,
        "../lora_kal_dp_sst2_model_best", "../lora_wo_sst2_model_best",
    )


def dadp_utility(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    args: list[Any] = [
        "--sigma", _method_value(settings, "dadp_icl", "sigma", 0.39),
        "--openai-model", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
        "--num-private-train", _method_value(settings, "dadp_icl", "num_private_train", 20),
        "--num-private-train-splits", _method_value(settings, "dadp_icl", "num_private_train_splits", 10),
        "--num-valid", _method_value(settings, "dadp_icl", "num_valid", 10),
        "--num-test", _method_value(settings, "dadp_icl", "num_test", 20),
        "--max-token-per-text", _method_value(settings, "dadp_icl", "max_token_per_text", 100),
        "--synth-seed", _method_value(settings, "dadp_icl", "synth_seed", 0),
        "--eval-seed", _method_value(settings, "dadp_icl", "eval_seed", 0),
    ]
    source = DADP_ROOT / "run_exp_sst2_loca_monitor.py"
    return [python_step(
        settings,
        "dadp_icl",
        "生成差分隐私示例并执行 SST-2 ICL 评估",
        source,
        args,
        run_dir,
        env=method_env(settings, DADP_ROOT),
        replacements={},
    )]


def dadp_nonprivate(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    source = DADP_ROOT / "run_exp_sst2_loca_monitor.py"
    return [python_step(
        settings,
        "dadp_icl",
        "关闭私有样本聚合，执行非隐私/零预算对照",
        source,
        [
            "--sigma", 0,
            "--openai-model", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
            "--num-private-train", 0,
            "--num-private-train-splits", 0,
            "--num-valid", _method_value(settings, "dadp_icl", "num_valid", 10),
            "--num-test", _method_value(settings, "dadp_icl", "num_test", 20),
        ],
        run_dir,
        env=method_env(settings, DADP_ROOT),
        replacements={},
    )]


def dadp_direct_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    source = DADP_ROOT / "direct_attack" / "member_test.py"
    return [python_step(
        settings,
        "dadp_icl",
        "复核已有前缀攻击输出与私有示例的重合率",
        source,
        [],
        run_dir,
        env=method_env(settings, source.parent),
        replacements={
            'json_path = "synthetic_text_results.json"': f'json_path = r"{DADP_ROOT / "direct_attack" / "synthetic_text_results.json"}"',
            'attack_path = "outputs/custom_prefix_samples_non.txt"': f'attack_path = r"{DADP_ROOT / "direct_attack" / "outputs" / "custom_prefix_samples.txt"}"',
        },
    )]


def dadp_direct_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    prompt_data = _as_path(_method_value(
        settings, "dadp_icl", "prompt_data_path", DADP_ROOT / "synthetic_text_results.json"
    ))
    attack_samples = max(1, int(_method_value(settings, "dadp_icl", "attack_samples", 20)))
    attack_root = DADP_ROOT / "direct_attack"
    source = attack_root / "invade_prefix_with_custom.py"
    member_source = attack_root / "member_test.py"
    base_model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    score_model = str(_common_value(settings, "model_13b", "meta-llama/Llama-2-13b-hf"))
    steps: list[Step] = []
    for label, text_field in (("protected", "synthetic_text"), ("baseline", "original_text")):
        output = run_dir / f"{label}_custom_prefix_samples.txt"
        prompt_line = f'prompt_line = f"Text:{{item[\'{text_field}\']}}.Label:{{item[\'label\']}}\\n"'
        generation_replacements = _present_replacements(source, {
            "./synthetic_text_results.json": _source_string(prompt_data),
            "meta-llama/Llama-2-7b-hf": _source_string(base_model),
            "meta-llama/Llama-2-13b-hf": _source_string(score_model),
            'prompt_line = f"Text:{item[\'synthetic_text\']}.Label:{item[\'label\']}\\n"': prompt_line,
        })
        member_replacements = _present_replacements(member_source, {
            'json_path = "synthetic_text_results.json"': f'json_path = r"{_source_string(prompt_data)}"',
            'attack_path = "outputs/custom_prefix_samples_non.txt"': f'attack_path = r"{_source_string(output)}"',
        })
        steps.extend([
            python_step(
                settings,
                "dadp_icl",
                f"按原始说明用{('合成' if label == 'protected' else '原始')}示例提示执行前缀抽取",
                source,
                [
                    "--use-custom-prefixes",
                    "--custom-prefix-file", attack_root / "example_content.txt",
                    "--N", attack_samples,
                    "--batch-size", min(4, attack_samples),
                    "--outfile", output,
                ],
                run_dir,
                env=method_env(settings, attack_root, DADP_ROOT),
                replacements=generation_replacements,
            ),
            python_step(
                settings,
                "dadp_icl",
                f"按原始 member_test 统计{('合成提示' if label == 'protected' else '原始提示')}攻击命中率",
                member_source,
                [],
                run_dir,
                env=method_env(settings, attack_root),
                replacements=member_replacements,
            ),
        ])
    return steps


def _csv_attack_eval(settings: dict[str, Any], run_dir: Path, method_id: str, attack_root: Path) -> list[Step]:
    source = attack_root / "evaluate_attack_metrics.py"
    return [python_step(
        settings,
        method_id,
        "复核已有间接成员推断 CSV（AUC/TPR/FNR）",
        source,
        ["--file", attack_root / "attack_results_bert_combined_noise.csv", "--ratio", 0.02],
        run_dir,
        env=method_env(settings, attack_root),
        replacements=_present_replacements(source, {"plt.show()": "plt.close()"}),
    )]


def _native_example_indirect_attack(
    settings: dict[str, Any],
    run_dir: Path,
    method_id: str,
    source: Path,
    member_json: Path,
    *,
    member_literal: str,
    checkpoint: Path | None = None,
) -> list[Step]:
    """Run the original DADP-ICL/PrivSV neighbour scorer for members and non-members."""
    model = str(_common_value(settings, "model_7b_chat", "meta-llama/Llama-2-7b-chat-hf"))
    sample_count = max(1, int(_method_value(settings, method_id, "native_attack_samples", 20)))
    neighbor_count = max(1, int(_method_value(settings, method_id, "native_attack_neighbors", 10)))
    visible_gpus = [item.strip() for item in str(_common_value(settings, "gpu", "0")).split(",") if item.strip()]
    search_gpu = 1 if len(visible_gpus) > 1 else 0
    candidates = {
        "meta-llama/Llama-2-7b-chat-hf": _source_string(model),
        member_literal: _source_string(member_json),
        'dataset = load_dataset("glue", "sst2")': (
            'dataset = load_dataset("csv", '
            f'data_files=r"{_source_string(SHARED_SST2_ROOT / "test.tsv")}", delimiter="\\t")'
        ),
        'texts = dataset["test"]["sentence"]': 'texts = dataset["train"]["sentence"]',
        "batch_size = 500": f"batch_size = {sample_count}",
        "num_neighbors = 10": f"num_neighbors = {neighbor_count}",
        "'cuda:3'": f"'cuda:{search_gpu}'",
    }
    if method_id == "privsv":
        candidates["'cuda:2'"] = "'cuda:0'"
        if checkpoint is None:
            raise ValueError("PrivSV 原生间接攻击缺少 steering checkpoint")
        candidates['"private_steering_vectors_llama_7b_hf.pt"'] = f'r"{_source_string(checkpoint)}"'
    replacements = _present_replacements(source, candidates)
    env_paths = [source.parent]
    if method_id == "privsv":
        env_paths.append(PRIVSV_ROOT)
    member_csv = run_dir / "attack_results_other_bert_0_noise.csv"
    nonmember_csv = run_dir / "attack_results_sst2_bert_0_noise.csv"
    return [
        python_step(
            settings,
            method_id,
            "原生邻居攻击：生成成员得分",
            source,
            ["--proc-id", 0, "--model", "bert", "--dataset", "other"],
            run_dir,
            env=method_env(settings, *env_paths),
            replacements=replacements,
        ),
        python_step(
            settings,
            method_id,
            "原生邻居攻击：生成非成员得分",
            source,
            ["--proc-id", 0, "--model", "bert", "--dataset", "sst2"],
            run_dir,
            env=method_env(settings, *env_paths),
            replacements=replacements,
        ),
        python_step(
            settings,
            method_id,
            "评估本次原生邻居成员推断结果",
            NATIVE_CSV_ATTACK_EVALUATOR,
            [
                "--member-csv", member_csv,
                "--nonmember-csv", nonmember_csv,
                "--method", method_id,
                "--target-fprs", 0.01, 0.001,
                "--output", run_dir / "native_indirect_attack.json",
            ],
            run_dir,
            env=method_env(settings, *env_paths),
        ),
    ]


def dadp_indirect_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _csv_attack_eval(settings, run_dir, "dadp_icl", DADP_ROOT / "Indirect_attack")


def _combine_example_attack(
    settings: dict[str, Any],
    run_dir: Path,
    method_id: str,
    attack_root: Path,
    label: str,
    member_csv: Path,
    nonmember_csv: Path,
    combined_csv: Path,
) -> list[Step]:
    """Use the repository's combine/evaluate scripts for one protected or baseline pair."""
    combine_source = attack_root / "combine.py"
    evaluate_source = attack_root / "evaluate_attack_metrics.py"
    combine_replacements = _present_replacements(combine_source, {
        'pd.read_csv("./attack_results_other_bert_0_noise.csv")': f'pd.read_csv(r"{_source_string(member_csv)}")',
        'pd.read_csv("./attack_results_sst2_bert_0_noise.csv")': f'pd.read_csv(r"{_source_string(nonmember_csv)}")',
        'df_all.to_csv("attack_results_bert_combined_noise.csv", index=False)': (
            f'df_all.to_csv(r"{_source_string(combined_csv)}", index=False)'
        ),
        'print(f"✅ 已合并成员+非成员文件，保存为 attack_results_bert_combined_noise.csv")': (
            f'print("✅ 当前评估：{label}；已合并成员+非成员文件，保存为：{_source_string(combined_csv)}")'
        ),
    })
    return [
        python_step(
            settings, method_id, f"按原始逻辑合并{label}成员与非成员得分",
            combine_source, [], run_dir,
            env=method_env(settings, attack_root), replacements=combine_replacements,
        ),
        python_step(
            settings, method_id, f"按原始逻辑计算{label}攻击性能指标",
            evaluate_source, ["--file", combined_csv, "--ratio", 0.1], run_dir,
            env=method_env(settings, attack_root),
            replacements=_present_replacements(evaluate_source, {"plt.show()": "plt.close()"}),
        ),
    ]


def dadp_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    attack_root = DADP_ROOT / "Indirect_attack"
    source = DADP_ROOT / "Indirect_attack" / "neighbour_mia.py"
    member_json = _as_path(_method_value(
        settings, "dadp_icl", "prompt_data_path", DADP_ROOT / "synthetic_text_results.json"
    ))
    model = str(_common_value(settings, "model_7b_chat", "meta-llama/Llama-2-7b-chat-hf"))
    sample_count = max(1, int(_method_value(settings, "dadp_icl", "native_attack_samples", 20)))
    neighbor_count = max(1, int(_method_value(settings, "dadp_icl", "native_attack_neighbors", 10)))
    visible_gpus = [item.strip() for item in str(_common_value(settings, "gpu", "0")).split(",") if item.strip()]
    search_gpu = 1 if len(visible_gpus) > 1 else 0
    outputs = {
        ("protected", "other"): run_dir / "attack_results_other_bert_0_noise.csv",
        ("protected", "sst2"): run_dir / "attack_results_sst2_bert_0_noise.csv",
        ("baseline", "other"): run_dir / "attack_results_other_bert_0.csv",
        ("baseline", "sst2"): run_dir / "attack_results_sst2_bert_0.csv",
    }
    steps: list[Step] = []
    for scope, field in (("protected", "synthetic_text"), ("baseline", "original_text")):
        for dataset in ("other", "sst2"):
            output = outputs[(scope, dataset)]
            replacements = _present_replacements(source, {
                "meta-llama/Llama-2-7b-chat-hf": _source_string(model),
                "./synthetic_text_results.json": _source_string(member_json),
                'dataset = load_dataset("glue", "sst2")': (
                    'dataset = load_dataset("csv", '
                    f'data_files=r"{_source_string(SHARED_SST2_ROOT / "test.tsv")}", delimiter="\\t")'
                ),
                'texts = dataset["test"]["sentence"]': 'texts = dataset["train"]["sentence"]',
                "batch_size = 500": f"batch_size = {sample_count}",
                "num_neighbors = 10": f"num_neighbors = {neighbor_count}",
                "'cuda:3'": f"'cuda:{search_gpu}'",
                'prompt_line = f"Text:{item[\'synthetic_text\']}.Label:{item[\'label\']}\\n"': (
                    f'prompt_line = f"Text:{{item[\'{field}\']}}.Label:{{item[\'label\']}}\\n"'
                ),
                'output_path = f"attack_results_{args.dataset}_{args.model}_{args.proc_id}_noise.csv"': (
                    f'output_path = r"{_source_string(output)}"'
                ),
            })
            steps.append(python_step(
                settings, "dadp_icl",
                f"原始邻居攻击：{('合成' if scope == 'protected' else '原始')}提示 / "
                f"{('成员' if dataset == 'other' else '非成员')}",
                source, ["--proc-id", 0, "--model", "bert", "--dataset", dataset], run_dir,
                env=method_env(settings, attack_root), replacements=replacements,
            ))
    protected_combined = run_dir / "attack_results_bert_combined_noise.csv"
    baseline_combined = run_dir / "attack_results_bert_combined.csv"
    steps += _combine_example_attack(
        settings, run_dir, "dadp_icl", attack_root, "所提出方法",
        outputs[("protected", "other")], outputs[("protected", "sst2")], protected_combined,
    )
    steps += _combine_example_attack(
        settings, run_dir, "dadp_icl", attack_root, "对比方法",
        outputs[("baseline", "other")], outputs[("baseline", "sst2")], baseline_combined,
    )
    return steps


def privsv_utility(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    layers = _method_value(settings, "privsv", "layers", [11, 12, 13, 14, 15])
    source = PRIVSV_ROOT / "run.py"
    return [python_step(
        settings,
        "privsv",
        "训练私有 steering vector 并评估 SST-2",
        source,
        [
            "--model", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
            "--dataset", _method_value(settings, "privsv", "dataset", "sst2"),
            "--layers", *layers,
            "--noise_multiplier", _method_value(settings, "privsv", "noise_multiplier", 0.06),
            "--clip", _method_value(settings, "privsv", "clip", 18),
        ],
        run_dir,
        env=method_env(settings, PRIVSV_ROOT),
        replacements=_present_replacements(source, {
            "device = 'cuda:3' if torch.cuda.is_available() else 'cpu'": "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'",
            'json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", \'r\'))': f'json.load(open(r"{_source_string(PRIVSV_ROOT / "datasets" / "generate")}" + f"/{{dataset}}/generate_dataset.json", \'r\'))',
            'json.load(open(f"./datasets/test/{dataset}/test_dataset_ab.json", \'r\'))': f'json.load(open(r"{_source_string(PRIVSV_ROOT / "datasets" / "test")}" + f"/{{dataset}}/test_dataset_ab.json", \'r\'))',
            "    pca=False\n)": "    pca=False,\n    clip=args.clip\n)",
        }),
    )]


def privsv_ablation(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    layers = _method_value(settings, "privsv", "layers", [11, 12, 13, 14, 15])
    source = PRIVSV_ROOT / "run.py"
    return [python_step(
        settings,
        "privsv",
        "关闭 steering vector 噪声的效用对照",
        source,
        [
            "--model", _common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"),
            "--dataset", _method_value(settings, "privsv", "dataset", "sst2"),
            "--layers", *layers,
            "--noise_multiplier", 0,
            "--clip", _method_value(settings, "privsv", "clip", 18),
        ],
        run_dir,
        env=method_env(settings, PRIVSV_ROOT),
        replacements=_present_replacements(source, {
            "device = 'cuda:3' if torch.cuda.is_available() else 'cpu'": "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'",
            'json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", \'r\'))': f'json.load(open(r"{_source_string(PRIVSV_ROOT / "datasets" / "generate")}" + f"/{{dataset}}/generate_dataset.json", \'r\'))',
            'json.load(open(f"./datasets/test/{dataset}/test_dataset_ab.json", \'r\'))': f'json.load(open(r"{_source_string(PRIVSV_ROOT / "datasets" / "test")}" + f"/{{dataset}}/test_dataset_ab.json", \'r\'))',
            "    pca=False\n)": "    pca=False,\n    clip=args.clip\n)",
        }),
    )]


def privsv_direct_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    attack_root = PRIVSV_ROOT / "direct_attack"
    return [python_step(
        settings,
        "privsv",
        "复核已有抽取样本与 PrivSV 训练示例的重合率",
        attack_root / "member_test-sv.py",
        [
            "--dataset", PRIVSV_ROOT / "datasets" / "generate" / "sst2" / "generate_dataset.json",
            "--attack-files", attack_root / "outputs" / "top_n_samples-sv.txt",
        ],
        run_dir,
        env=method_env(settings, attack_root),
    )]


def privsv_direct_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    attack_root = PRIVSV_ROOT / "direct_attack"
    baseline_output = run_dir / "top_n_samples.txt"
    protected_output = run_dir / "top_n_samples-sv.txt"
    model = str(_common_value(settings, "model_7b", "meta-llama/Llama-2-7b-hf"))
    baseline_adapter = str(_method_value(
        settings, "privsv", "baseline_adapter_path", "Styxxxx/llama2_7b_lora-sst2"
    ))
    attack_samples = max(1, int(_method_value(settings, "privsv", "attack_samples", 20)))
    attack_batch_size = min(6, attack_samples)
    visible_gpus = [item.strip() for item in str(_common_value(settings, "gpu", "0")).split(",") if item.strip()]
    second_device = "torch.device('cuda:1' if torch.cuda.is_available() and gpu_count > 1 else device1)" if len(visible_gpus) > 1 else "torch.device('cuda:0' if torch.cuda.is_available() and gpu_count > 1 else device1)"
    baseline_source = attack_root / "extract_topn.py"
    protected_source = attack_root / "extract_topn-sv.py"
    steering_checkpoint = _as_path(_method_value(
        settings,
        "privsv",
        "steering_checkpoint_path",
        PRIVSV_ROOT / "private_steering_vectors_llama_7b_hf.pt",
    ))
    baseline_replacements = _present_replacements(baseline_source, {
        "device1 = torch.device('cuda:1')": "device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        "device2 = torch.device('cuda:2')": second_device,
        '"meta-llama/Llama-2-7b-hf"': f'"{_source_string(model)}"',
        '"Styxxxx/llama2_7b_lora-sst2"': f'"{_source_string(baseline_adapter)}"',
    })
    protected_replacements = _present_replacements(protected_source, {
        'torch.device(\'cuda:2\' if torch.cuda.is_available() else \'cpu\')': 'torch.device(\'cuda:0\' if torch.cuda.is_available() else \'cpu\')',
        'torch.device(\'cuda:3\' if torch.cuda.is_available() and gpu_count > 1 else device1)': second_device,
        '"meta-llama/Llama-2-7b-hf"': f'"{_source_string(model)}"',
        'torch.load("mean_steering_vectors_llama_7b_hf.pt", map_location="cpu")': f'torch.load(r"{_source_string(steering_checkpoint)}", map_location="cpu")',
    })
    member_source = attack_root / "member_test.py"
    member_replacements = _present_replacements(member_source, {
        'tsv_path = "../../privacy_tools/datasets/sst2/train.tsv"': (
            f'tsv_path = r"{_source_string(SHARED_SST2_ROOT / "train.tsv")}"'
        ),
        'attack_path = "./outputs/top_n_samples.txt"': (
            f'attack_path = r"{_source_string(baseline_output)}"'
        ),
    })
    return [
        python_step(
            settings,
            "privsv",
            "按原始说明执行未保护模型前缀抽取",
            baseline_source,
            ["--N", attack_samples, "--batch_size", attack_batch_size, "--outfile", baseline_output],
            run_dir,
            env=method_env(settings, attack_root),
            replacements=baseline_replacements,
        ),
        python_step(
            settings,
            "privsv",
            "按原始说明执行 PrivSV 前缀抽取",
            protected_source,
            ["--N", attack_samples, "--batch_size", attack_batch_size, "--outfile", protected_output],
            run_dir,
            env=method_env(settings, attack_root, PRIVSV_ROOT),
            replacements=protected_replacements,
        ),
        python_step(
            settings,
            "privsv",
            "按原始 member_test 统计未保护攻击命中率",
            member_source,
            [],
            run_dir,
            env=method_env(settings, attack_root),
            replacements=member_replacements,
        ),
        python_step(
            settings,
            "privsv",
            "按原始 member_test-sv 统计 PrivSV 攻击命中率",
            attack_root / "member_test-sv.py",
            ["--dataset", PRIVSV_ROOT / "datasets" / "generate" / "sst2" / "generate_dataset.json", "--attack-files", protected_output],
            run_dir,
            env=method_env(settings, attack_root),
        ),
    ]


def privsv_indirect_existing(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    return _csv_attack_eval(settings, run_dir, "privsv", PRIVSV_ROOT / "Indirect_attack")


def privsv_indirect_attack(settings: dict[str, Any], run_dir: Path) -> list[Step]:
    attack_root = PRIVSV_ROOT / "Indirect_attack"
    source = attack_root / "neighbour_mia_sv.py"
    member_json = _as_path(_method_value(
        settings,
        "privsv",
        "native_member_data_path",
        PRIVSV_ROOT / "Indirect_attack" / "sst2_train.json",
    ))
    checkpoint = _as_path(_method_value(
        settings,
        "privsv",
        "steering_checkpoint_path",
        PRIVSV_ROOT / "private_steering_vectors_llama_7b_hf.pt",
    ))
    model = str(_common_value(settings, "model_7b_chat", "meta-llama/Llama-2-7b-chat-hf"))
    sample_count = max(1, int(_method_value(settings, "privsv", "native_attack_samples", 20)))
    neighbor_count = max(1, int(_method_value(settings, "privsv", "native_attack_neighbors", 10)))
    multiplier = float(_method_value(settings, "privsv", "steering_multiplier", 2.0))
    visible_gpus = [item.strip() for item in str(_common_value(settings, "gpu", "0")).split(",") if item.strip()]
    search_gpu = 1 if len(visible_gpus) > 1 else 0
    outputs = {
        ("protected", "other"): run_dir / "attack_results_other_bert_0_noise.csv",
        ("protected", "sst2"): run_dir / "attack_results_sst2_bert_0_noise.csv",
        ("baseline", "other"): run_dir / "attack_results_other_bert_0.csv",
        ("baseline", "sst2"): run_dir / "attack_results_sst2_bert_0.csv",
    }
    steps: list[Step] = []
    for scope, steering_value in (("protected", multiplier), ("baseline", 0.0)):
        for dataset in ("other", "sst2"):
            output = outputs[(scope, dataset)]
            replacements = _present_replacements(source, {
                "meta-llama/Llama-2-7b-chat-hf": _source_string(model),
                "./sst2_train.json": _source_string(member_json),
                'dataset = load_dataset("glue", "sst2")': (
                    'dataset = load_dataset("csv", '
                    f'data_files=r"{_source_string(SHARED_SST2_ROOT / "test.tsv")}", delimiter="\\t")'
                ),
                'texts = dataset["test"]["sentence"]': 'texts = dataset["train"]["sentence"]',
                "batch_size = 500": f"batch_size = {sample_count}",
                "num_neighbors = 10": f"num_neighbors = {neighbor_count}",
                "'cuda:3'": f"'cuda:{search_gpu}'",
                "'cuda:2'": "'cuda:0'",
                '"private_steering_vectors_llama_7b_hf.pt"': f'r"{_source_string(checkpoint)}"',
                "multiplier=2.0": f"multiplier={steering_value}",
                'output_path = f"attack_results_{args.dataset}_{args.model}_{args.proc_id}_noise.csv"': (
                    f'output_path = r"{_source_string(output)}"'
                ),
            })
            steps.append(python_step(
                settings, "privsv",
                f"原始邻居攻击：{('PrivSV' if scope == 'protected' else '未保护')} / "
                f"{('成员' if dataset == 'other' else '非成员')}",
                source, ["--proc-id", 0, "--model", "bert", "--dataset", dataset], run_dir,
                env=method_env(settings, attack_root, PRIVSV_ROOT), replacements=replacements,
            ))
    protected_combined = run_dir / "attack_results_bert_combined_noise.csv"
    baseline_combined = run_dir / "attack_results_bert_combined.csv"
    steps += _combine_example_attack(
        settings, run_dir, "privsv", attack_root, "所提出方法",
        outputs[("protected", "other")], outputs[("protected", "sst2")], protected_combined,
    )
    steps += _combine_example_attack(
        settings, run_dir, "privsv", attack_root, "对比方法",
        outputs[("baseline", "other")], outputs[("baseline", "sst2")], baseline_combined,
    )
    return steps


def embedding_methods() -> dict[str, Method]:
    return {
        "cliptext": Method(
            "cliptext",
            "ClipText",
            "对输入词嵌入构造 top-k 候选集，并用指数机制替换词；可选择保留停用词。",
            CLIP_ROOT,
            [CLIP_SST / "main.py", CLIP_ROOT / "dp" / "cliptext.py", CLIP_ATTACK / "neighbour_attack.py", CLIP_ATTACK / "neighbour_attack_wo.py", ATTACK_EVALUATOR],
            [
                Experiment("utility", "本微调方法", "在 ClipText 脱敏文本上进行微调，输出 Acc、内存、运行时间和吞吐率。", clip_llama_utility),
                Experiment("baseline", "对比微调方法", "在原始 SST-2 文本上运行相同 LoRA 微调，输出对比 Acc 与效率指标。", clip_baseline),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", clip_prefix_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", clip_indirect_attack),
            ],
            warnings=["直接评估和间接评估需要提供 ClipText 的受保护模型；间接评估还需要非保护对比模型。"],
            path_name="ClipText",
        ),
        "ptcds": Method(
            "ptcds",
            "PTCDS",
            "根据当前数据集进行模型拆分的嵌入扰动微调。",
            PTCDS_ROOT,
            [
                PTCDS_ROOT / "finetune_llama2_sst2_lora_data.py",
                PTCDS_ATTACK / "extraction_temperature_decay.py",
                PTCDS_ATTACK / "extraction_temperature_decay_base.py",
                PTCDS_ATTACK / "neighbour_attack.py",
                PTCDS_ATTACK / "neighbour_attack_wo.py",
                PTCDS_ATTACK / "all_scores_sst2_roberta_0_train.pkl",
            ],
            [
                Experiment("utility", "本微调方法", "运行 PTCDS 微调，输出 Acc、内存、运行时间和吞吐率。", ptcds_train),
                Experiment("baseline", "对比微调方法", "在原始 SST-2 文本上运行 LoRA 对比微调，输出对比 Acc 与效率指标。", ptcds_baseline),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", ptcds_prefix_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", ptcds_indirect_attack),
            ],
            warnings=["直接评估和间接评估需要提供 PTCDS 受保护模型与未保护 base 对比模型的 adapter 路径。"],
            path_name="PTCDS",
        ),
    }


def gradient_methods() -> dict[str, Method]:
    return {
        "kal_dp": Method(
            "kal_dp",
            "Kal-DP",
            "对裁剪后的带噪梯度进行 Kalman 状态估计/降噪",
            KAL_ROOT,
            [KAL_ROOT / "kal_dp.py", KAL_ROOT / "dpsgd.py", KAL_ATTACK / "neighbour_attack.py", KAL_ATTACK / "neighbour_attack_wo.py"],
            [
                Experiment("utility", "本微调方法", "运行 Kal-DP 微调，输出 Acc、内存、运行时间和吞吐率。", kal_train),
                Experiment("baseline", "对比微调方法", "运行相同隐私预算下的 DP-SGD 对比微调，输出 Acc、内存、运行时间和吞吐率。", dpsgd_train),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", kal_prefix_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", kal_indirect_attack),
            ],
            path_name="Kal-DP",
        ),
        "zo_dpga": Method(
            "zo_dpga",
            "ZO-DPGA",
            "用零阶有限差分方向导数完成带隐私保护的更新，避免保存反向传播参数。",
            ZO_ROOT,
            [ZO_ROOT / "run_formal_experiment.py", ZO_REPRODUCTION, ZO_ROOT / "tests" / "validate_zo_math.py"],
            [
                Experiment("utility", "本微调方法", "运行 ZO-DPGA 微调，输出 Acc、内存、运行时间和吞吐率。", zo_reproduce),
                Experiment("baseline", "对比微调方法", "运行 LoRA + MeZO 对比微调，输出对比 Acc 与效率指标。", zo_baseline),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", zo_prefix_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", zo_indirect_attack),
            ],
            path_name="ZO-DPGA",
        ),
    }


def example_methods() -> dict[str, Method]:
    return {
        "dadp_icl": Method(
            "dadp_icl",
            "DADP-ICL",
            "把私有示例分组聚合，再加噪生成合成示例，用于上下文学习隐式微调。",
            DADP_ROOT,
            [DADP_ROOT / "run_exp_sst2_loca_monitor.py", DADP_ROOT / "direct_attack" / "member_test.py", DADP_ROOT / "Indirect_attack" / "neighbour_mia.py"],
            [
                Experiment("utility", "本微调方法", "运行 DADP-ICL 私有示例生成与隐式微调，输出 Acc、内存、运行时间和吞吐率。", dadp_utility),
                Experiment("baseline", "对比微调方法", "关闭私有聚合并进行隐式微调，输出对比 Acc 与效率指标。", dadp_nonprivate),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", dadp_direct_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", dadp_indirect_attack),
            ],
            path_name="DADP-ICL",
        ),
        "privsv": Method(
            "privsv",
            "PrivSV",
            "从正负示例中学习引导向量，经裁剪和拉普拉斯噪声保护后进行隐式微调。",
            PRIVSV_ROOT,
            [PRIVSV_ROOT / "run.py", PRIVSV_ROOT / "steering_wrapper.py", PRIVSV_ROOT / "direct_attack" / "member_test-sv.py", PRIVSV_ROOT / "Indirect_attack" / "neighbour_mia_sv.py"],
            [
                Experiment("utility", "本微调方法", "训练私有引导向量并评估，输出 Acc、内存、运行时间和吞吐率。", privsv_utility),
                Experiment("baseline", "对比微调方法", "运行无噪声引导向量对比方法，输出对比 Acc 与效率指标。", privsv_ablation),
                Experiment("direct_attack", "直接评估", "执行训练数据抽取评估并保存攻击命中指标。", privsv_direct_attack),
                Experiment("indirect_attack", "间接评估", "执行邻居成员推断评估并保存攻击性能指标。", privsv_indirect_attack),
            ],
            path_name="PrivSV",
        ),
    }
