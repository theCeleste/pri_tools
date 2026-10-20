from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .bootstrap import (
    GLOVE_TARGET,
    clip_mapping_available,
    check_cuda,
    check_imports,
    copy_glove,
    download_glove,
    environment_plan,
    ensure_nltk_stopwords,
)
from .catalog import (
    EVALUATION_METHOD_GROUPS,
    FINETUNE_METHOD_GROUPS,
    embedding_methods,
    example_methods,
    gradient_methods,
    kal_no_perturbation_direct,
    kal_no_perturbation_indirect,
    dadp_no_perturbation_direct,
    dadp_no_perturbation_indirect,
)
from .core import (
    Experiment,
    Method,
    choose_from_menu,
    doctor,
    get_nested,
    list_catalog,
    load_settings,
    project_path,
    run_experiment,
)


METHODS: dict[str, Method] = {
    "dadp_icl": example_methods()["dadp_icl"],
    "privsv": example_methods()["privsv"],
    "cliptext": embedding_methods()["cliptext"],
    "ptcds": embedding_methods()["ptcds"],
    "kal_dp": gradient_methods()["kal_dp"],
    "zo_dpga": gradient_methods()["zo_dpga"],
}

CATEGORIES = {
    "cliptext": "embedding",
    "ptcds": "embedding",
    "kal_dp": "gradient",
    "zo_dpga": "gradient",
    "dadp_icl": "example",
    "privsv": "example",
}

METHOD_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "example": ("示例扰动微调", ("dadp_icl", "privsv")),
    "embedding": ("嵌入扰动微调", ("cliptext", "ptcds")),
    "gradient": ("梯度扰动微调", ("kal_dp", "zo_dpga")),
}


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    kind: str = "str"
    hint: str = ""


@dataclass(frozen=True)
class TaskSelection:
    task_kind: str
    method_id: str
    experiment_id: str
    perturbation_group: str
    evaluation_type: str | None = None
    finetune_type: str | None = None
    model: str | None = None
    dataset: str | None = None
    unsupported: bool = False
    evaluation_model: str | None = None
    evaluation_model_title: str | None = None
    backend_method_id: str | None = None
    display_method_title: str | None = None
    backend: str | None = None


PYTHON = PromptSpec("", "Python 解释器", "str", "直接回车使用当前激活环境")
MODEL = PromptSpec("common.model_7b", "基础模型路径或 Hugging Face 名称", "str")
MODEL_CHAT = PromptSpec("common.model_7b_chat", "对话基础模型路径或 Hugging Face 名称", "str")
MODEL_13B = PromptSpec("common.model_13b", "直接攻击评分模型路径或 Hugging Face 名称", "str")
GPU = PromptSpec("common.gpu", "GPU 编号", "str", "单卡如 0；PrivSV 完整攻击推荐两卡，如 0,1")

EPS_CLIP = PromptSpec("methods.cliptext.epsilon", "隐私预算 epsilon", "float")
TOP_K = PromptSpec("methods.cliptext.top_k", "候选词 top-k", "int")
STOP_WORDS = PromptSpec("methods.cliptext.save_stop_words", "是否保留停用词", "bool")
CLIP_BATCH = PromptSpec("methods.cliptext.batch_size", "batch size", "int")
CLIP_EPOCHS = PromptSpec("methods.cliptext.epochs", "训练轮数 epochs", "int")

PT_TRAIN_DATA = PromptSpec("methods.ptcds.train_data_path", "PTCDS 扰动后训练集路径", "str")

KAL_EPS = PromptSpec("methods.kal_dp.epsilon", "隐私预算 epsilon", "float")
KAL_DELTA = PromptSpec("methods.kal_dp.delta", "隐私参数 delta", "float")
KAL_BATCH = PromptSpec("methods.kal_dp.batch_size", "batch size", "int")
KAL_EPOCHS = PromptSpec("methods.kal_dp.epochs", "训练轮数 epochs", "int")
KAL_NORM = PromptSpec("methods.kal_dp.max_grad_norm", "梯度裁剪阈值", "float")

DADP_SIGMA = PromptSpec("methods.dadp_icl.sigma", "高斯噪声 sigma", "float")
DADP_TRAIN = PromptSpec("methods.dadp_icl.num_private_train", "私有训练示例数", "int")
DADP_SPLITS = PromptSpec("methods.dadp_icl.num_private_train_splits", "私有训练分组数", "int")
DADP_VALID = PromptSpec("methods.dadp_icl.num_valid", "验证样本数", "int")
DADP_TEST = PromptSpec("methods.dadp_icl.num_test", "测试样本数", "int")
DADP_TOKENS = PromptSpec("methods.dadp_icl.max_token_per_text", "每条文本最大 token 数", "int")

PRIV_LAYERS = PromptSpec("methods.privsv.layers", "注入层编号", "int_list", "逗号或空格分隔")
PRIV_NOISE = PromptSpec("methods.privsv.noise_multiplier", "噪声系数", "float")
PRIV_CLIP = PromptSpec("methods.privsv.clip", "向量裁剪阈值", "int")


def _attack_specs(method_id: str, *, adapter: bool = True) -> list[PromptSpec]:
    specs = [MODEL, GPU]
    if adapter:
        specs += [
            PromptSpec(f"methods.{method_id}.adapter_path", "所提出方法 adapter 路径", "str"),
            PromptSpec(f"methods.{method_id}.baseline_adapter_path", "对比方法 adapter 路径", "str"),
        ]
    specs.append(PromptSpec(f"methods.{method_id}.attack_samples", "攻击生成样本数", "int"))
    return specs


def _indirect_attack_specs(method_id: str, *, protection: str) -> list[PromptSpec]:
    specs = [
        MODEL,
        GPU,
        PromptSpec("common.mask_model", "邻居生成掩码模型路径或 Hugging Face 名称", "str"),
    ]
    if protection == "adapter":
        specs.append(PromptSpec(f"methods.{method_id}.adapter_path", "受保护 adapter 路径", "str"))
    elif protection == "prompt":
        specs += [
            PromptSpec("methods.dadp_icl.prompt_data_path", "DADP-ICL 受保护示例 JSON 路径", "str"),
            PromptSpec("methods.dadp_icl.prompt_examples", "加入攻击上下文的受保护示例数", "int"),
        ]
    elif protection == "steering":
        specs += [
            PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 引导向量路径", "str"),
            PromptSpec("methods.privsv.steering_multiplier", "steering 注入倍数", "float"),
        ]
    specs += [
        PromptSpec(f"methods.{method_id}.indirect_samples", "成员与非成员各抽取的样本数", "int"),
        PromptSpec(f"methods.{method_id}.indirect_neighbors", "每条文本生成的邻居数", "int"),
    ]
    return specs


def _native_adapter_indirect_specs(method_id: str) -> list[PromptSpec]:
    return [
        MODEL,
        GPU,
        PromptSpec("common.mask_model", "原生邻居生成掩码模型路径或 Hugging Face 名称", "str"),
        PromptSpec(f"methods.{method_id}.adapter_path", "受保护 adapter 路径", "str"),
        PromptSpec(f"methods.{method_id}.baseline_adapter_path", "非保护基线 adapter 路径", "str"),
        PromptSpec(f"methods.{method_id}.native_attack_samples", "成员与非成员各处理的样本数", "int"),
    ]


def prompt_specs(method_id: str, experiment_id: str) -> list[PromptSpec]:
    """Return only settings that the selected experiment actually consumes."""
    lightweight = {"direct_existing", "indirect_existing", "validate", "summarize"}
    if experiment_id in lightweight:
        return []
    if method_id == "cliptext":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id)
        if experiment_id == "indirect_attack":
            return _native_adapter_indirect_specs(method_id)
        if experiment_id == "utility":
            return [MODEL, GPU, EPS_CLIP, TOP_K, STOP_WORDS, CLIP_BATCH, CLIP_EPOCHS]
        if experiment_id in {"baseline", "performance_baseline"}:
            return [MODEL, GPU, CLIP_BATCH, CLIP_EPOCHS]
        return []
    if method_id == "ptcds":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id)
        if experiment_id == "indirect_attack":
            return _native_adapter_indirect_specs(method_id)
        if experiment_id in {"baseline", "performance_baseline"}:
            return [MODEL, GPU]
        return [MODEL, GPU, PT_TRAIN_DATA]
    if method_id == "kal_dp":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id)
        if experiment_id == "indirect_attack":
            return _native_adapter_indirect_specs(method_id)
        return [MODEL, GPU, KAL_EPS, KAL_DELTA, KAL_BATCH, KAL_EPOCHS, KAL_NORM]
    if method_id == "zo_dpga":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id)
        if experiment_id == "indirect_attack":
            return _native_adapter_indirect_specs(method_id)
        return [MODEL, GPU]
    if method_id == "dadp_icl":
        if experiment_id == "direct_attack":
            return [
                MODEL,
                MODEL_13B,
                GPU,
                PromptSpec("methods.dadp_icl.prompt_data_path", "DADP-ICL 受保护示例 JSON 路径", "str"),
                PromptSpec("methods.dadp_icl.prompt_examples", "加入攻击上下文的受保护示例数", "int"),
                PromptSpec("methods.dadp_icl.attack_samples", "攻击生成样本数", "int"),
            ]
        if experiment_id == "indirect_attack":
            return [
                MODEL_CHAT,
                GPU,
                PromptSpec("methods.dadp_icl.prompt_data_path", "DADP-ICL 受保护示例 JSON 路径", "str"),
                PromptSpec("methods.dadp_icl.native_attack_samples", "成员与非成员各处理的样本数", "int"),
                PromptSpec("methods.dadp_icl.native_attack_neighbors", "每条文本使用的邻居数", "int"),
            ]
        if experiment_id in {"baseline", "performance_baseline"}:
            return [MODEL, GPU, DADP_VALID, DADP_TEST]
        return [MODEL, GPU, DADP_SIGMA, DADP_TRAIN, DADP_SPLITS, DADP_VALID, DADP_TEST, DADP_TOKENS]
    if method_id == "privsv":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id, adapter=False) + [
                PromptSpec("methods.privsv.baseline_adapter_path", "未保护对比 adapter 路径或 Hugging Face 名称", "str"),
                PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 引导向量路径", "str")
            ]
        if experiment_id == "indirect_attack":
            return [
                MODEL_CHAT,
                GPU,
                PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 引导向量路径", "str"),
                PromptSpec("methods.privsv.native_member_data_path", "PrivSV 成员样本 JSON 路径", "str"),
                PromptSpec("methods.privsv.native_attack_samples", "成员与非成员各处理的样本数", "int"),
                PromptSpec("methods.privsv.native_attack_neighbors", "每条文本使用的邻居数", "int"),
            ]
        specs = [MODEL, GPU, PRIV_LAYERS, PRIV_CLIP]
        if experiment_id == "utility":
            specs.insert(-1, PRIV_NOISE)
        return specs
    return []


def _set_nested(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = mapping
    keys = dotted.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _parse_answer(raw: str, kind: str) -> Any:
    if kind == "str":
        return raw
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        lowered = raw.lower()
        if lowered in {"y", "yes", "1", "true", "是"}:
            return True
        if lowered in {"n", "no", "0", "false", "否"}:
            return False
        raise ValueError("请输入 y 或 n")
    if kind == "int_list":
        parts = [part for part in re.split(r"[,、\s]+", raw) if part]
        if not parts:
            raise ValueError("至少输入一个层编号")
        return [int(part) for part in parts]
    raise ValueError(f"未知参数类型：{kind}")


def _ask_setting(settings: dict[str, Any], spec: PromptSpec) -> Any:
    default = get_nested(settings, spec.key)
    suffix = f"（{spec.hint}）" if spec.hint else ""
    while True:
        raw = input(f"  {spec.label}{suffix} [{_display_value(default)}]：").strip()
        if not raw:
            return default
        try:
            return _parse_answer(raw, spec.kind)
        except ValueError as exc:
            print(f"    输入无效：{exc}")


def _yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]：" if default else " [y/N]："
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "是"}:
            return True
        if raw in {"n", "no", "否"}:
            return False
        print("请输入 y 或 n。")


def _confirm_real_run() -> bool:
    """Require an explicit, retryable confirmation before a real experiment."""
    while True:
        confirm = input("方法运行可能耗时较长。请输入 run 确认；输入 0 取消：").strip().lower()
        if confirm == "run":
            return True
        if confirm in {"0", "q", "quit", "cancel", "取消"}:
            return False
        print("输入不正确，请重新输入小写 run；如需取消请输入 0。")


def _prepare_runtime(method_id: str, experiment_id: str, settings: dict[str, Any], dry_run: bool) -> bool:
    runtime_method = "dpsgd" if experiment_id == "performance_baseline" and method_id not in {"kal_dp", "zo_dpga"} else method_id
    runtime_experiment = "utility" if experiment_id == "performance_baseline" else experiment_id
    plan = environment_plan(runtime_method, runtime_experiment)
    python_profile = "kal_dp" if experiment_id == "performance_baseline" and method_id in {"kal_dp", "zo_dpga"} else runtime_method
    configured = str(get_nested(
        settings,
        f"python.{python_profile}",
        get_nested(settings, "python.default", sys.executable),
    ))
    selected = Path(configured).expanduser()
    _set_nested(settings, f"python.{python_profile}", configured)
    if dry_run:
        print(f"  查看最近历史结果使用默认配置的 Python环境：{configured}")
        print("  查看最近历史结果不会安装依赖。")
        return True
    if plan.level == "none":
        print(f"  {plan.explanation}")
        print(f"  使用默认配置的 Python环境：{configured}")
        return True
    print(f"  {plan.explanation}")
    ready, missing = check_imports(selected, plan.imports)
    if ready:
        print(f"  [OK] 所需依赖已就绪：{selected}")
        return True
    print(f"  [缺少依赖] {', '.join(missing)}")
    print("已停止：默认配置指定的 Python 环境缺少实验依赖；请修改配置后重试。")
    return False


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~", "\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _prepare_assets(
    method_id: str,
    experiment_id: str,
    settings: dict[str, Any],
    specs: list[PromptSpec],
    dry_run: bool,
    *,
    baseline_only: bool = False,
) -> bool:
    if dry_run:
        print("  查看最近历史结果只检查命令结构，不下载模型或数据。")
        return True

    mapping_ready = False
    if method_id == "cliptext" and experiment_id in {"sanitize", "utility_bert"}:
        epsilon = get_nested(settings, "methods.cliptext.epsilon", 10.0)
        top_k = get_nested(settings, "methods.cliptext.top_k", 20)
        mapping_ready = clip_mapping_available(epsilon, top_k)
        if mapping_ready:
            print(f"  [OK] 已有 epsilon={epsilon}, top-k={top_k} 的预计算映射；本次不需要 GloVe。")

    if method_id == "cliptext" and experiment_id in {"sanitize", "utility_bert"} and not mapping_ready and not GLOVE_TARGET.exists():
        print("\n[需要数据] ClipText 词替换需要 glove_840B-300d.txt。")
        print("官方压缩包约 2.03 GiB，解压后约 5.3 GiB，准备时至少需要约 9 GiB 空间。")
        choice = choose_from_menu(
            "请选择 GloVe 准备方式",
            [
                ("download", "自动从 Stanford NLP 下载并解压"),
                ("copy", "输入服务器上已有的 glove_840B-300d.txt 路径"),
            ],
        )
        if choice is None:
            return False
        try:
            if choice == "download":
                if not _yes_no("确认开始大文件下载吗？"):
                    return False
                download_glove()
            else:
                source = input("请输入现有 GloVe 文件完整路径：").strip()
                if not source:
                    return False
                copy_glove(Path(source))
        except (OSError, ValueError) as exc:
            print(f"GloVe 准备失败：{exc}")
            return False
        print(f"  [OK] GloVe：{GLOVE_TARGET}")

    if method_id == "cliptext" and experiment_id in {"sanitize", "utility_bert"}:
        selected_python = str(get_nested(settings, f"python.{method_id}", sys.executable))
        if ensure_nltk_stopwords(selected_python) != 0:
            print("NLTK stopwords 下载失败，ClipText 目前无法启动。")
            return False

    model_spec = next((spec for spec in specs if spec.key in {"common.model_7b", "common.model_7b_chat"}), None)
    if model_spec is not None:
        model = str(get_nested(settings, model_spec.key, ""))
        if _looks_like_local_path(model):
            resolved = project_path(model)
            if not resolved.exists():
                print(f"[缺少模型] 本地模型路径不存在：{resolved}")
                return False
        # elif model.startswith("meta-llama/") and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        #     print("\n[模型权限提示] Llama-2 通常需要先在 Hugging Face 接受许可。")
        #     print("如果模型已完整缓存在服务器，可不输入 token。token 只用于本次进程，不会写入结果文件。")
        #     if _yes_no("现在输入 Hugging Face token 吗？", default=False):
        #         token = getpass.getpass("HF token（输入不回显）：").strip()
        #         if token:
        #             os.environ["HF_TOKEN"] = token

    if experiment_id == "direct_attack" and method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"}:
        adapter_keys = [f"methods.{method_id}.baseline_adapter_path"] if baseline_only else [
            f"methods.{method_id}.adapter_path", f"methods.{method_id}.baseline_adapter_path"
        ]
        for key in adapter_keys:
            while True:
                raw = str(get_nested(settings, key, ""))
                resolved = project_path(raw)
                if raw and resolved.exists():
                    print(f"  [OK] adapter：{resolved}")
                    _set_nested(settings, key, str(resolved))
                    break
                label = "基线 adapter" if key.endswith("baseline_adapter_path") else "受保护 adapter"
                print(f"\n[缺少 {label}] 当前路径不存在：{resolved}")
                print("直接攻击必须加载已训练 adapter。可先运行对应训练实验，再把输出目录填在这里。")
                replacement = input(f"请输入已有{label}目录；输入 0 取消：").strip()
                if replacement == "0" or not replacement:
                    return False
                _set_nested(settings, key, replacement)

    if experiment_id == "indirect_attack" and method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"}:
        adapter_keys = [f"methods.{method_id}.baseline_adapter_path"] if baseline_only else [
            f"methods.{method_id}.adapter_path", f"methods.{method_id}.baseline_adapter_path"
        ]
        for key in adapter_keys:
            while True:
                raw = str(get_nested(settings, key, ""))
                resolved = project_path(raw)
                if raw and resolved.exists():
                    label = "非保护基线 adapter" if key.endswith("baseline_adapter_path") else "受保护 adapter"
                    print(f"  [OK] {label}：{resolved}")
                    _set_nested(settings, key, str(resolved))
                    break
                label = "非保护基线 adapter" if key.endswith("baseline_adapter_path") else "受保护 adapter"
                print(f"\n[缺少{label}] 当前路径不存在：{resolved}")
                print("原生间接攻击需要同时比较受保护模型和非保护基线模型。")
                replacement = input(f"请输入已有{label}目录；输入 0 取消：").strip()
                if replacement == "0" or not replacement:
                    return False
                _set_nested(settings, key, replacement)

    if method_id == "dadp_icl" and experiment_id in {"direct_attack", "indirect_attack"}:
        key = "methods.dadp_icl.prompt_data_path"
        while True:
            raw = str(get_nested(settings, key, ""))
            resolved = project_path(raw)
            if raw and resolved.is_file():
                print(f"  [OK] DADP-ICL 受保护示例：{resolved}")
                _set_nested(settings, key, str(resolved))
                break
            print(f"\n[缺少 DADP-ICL 受保护示例] 当前文件不存在：{resolved}")
            replacement = input("请输入 synthetic_text_results.json 路径；输入 0 取消：").strip()
            if replacement == "0" or not replacement:
                return False
            _set_nested(settings, key, replacement)

    if method_id == "privsv" and experiment_id in {"direct_attack", "indirect_attack"}:
        key = "methods.privsv.steering_checkpoint_path"
        while True:
            raw = str(get_nested(settings, key, ""))
            resolved = project_path(raw)
            if raw and resolved.is_file():
                print(f"  [OK] PrivSV 引导向量：{resolved}")
                _set_nested(settings, key, str(resolved))
                break
            print(f"\n[缺少 PrivSV 引导向量] 当前文件不存在：{resolved}")
            replacement = input("请输入 private steering checkpoint 路径；输入 0 取消：").strip()
            if replacement == "0" or not replacement:
                return False
            _set_nested(settings, key, replacement)

    if method_id == "privsv" and experiment_id == "indirect_attack":
        key = "methods.privsv.native_member_data_path"
        while True:
            raw = str(get_nested(settings, key, ""))
            resolved = project_path(raw)
            if raw and resolved.is_file():
                print(f"  [OK] PrivSV 成员样本：{resolved}")
                _set_nested(settings, key, str(resolved))
                break
            print(f"\n[缺少 PrivSV 成员样本] 当前文件不存在：{resolved}")
            replacement = input("请输入含 text 字段的成员样本 JSON；输入 0 取消：").strip()
            if replacement == "0" or not replacement:
                return False
            _set_nested(settings, key, replacement)

    if any(spec.key == "common.gpu" for spec in specs):
        selected_python = str(get_nested(settings, f"python.{method_id}", sys.executable))
        gpu = str(get_nested(settings, "common.gpu", "0"))
        ok, detail = check_cuda(selected_python, gpu)
        print(f"  [{'OK' if ok else 'FAIL'}] GPU：{detail}")
        if not ok:
            print("当前实验需要 GPU。请检查显卡驱动、CUDA 版 PyTorch 或 GPU 编号后重试。")
            return False
    return True


def _choose_model_and_dataset(
    settings: dict[str, Any],
    *,
    method_id: str,
    experiment_id: str,
    choose_dataset: bool = True,
) -> tuple[str | None, str | None]:
    """Select only JSON-defined model/dataset values; custom input is disabled."""
    uses_chat_model = experiment_id == "indirect_attack" and method_id in {"dadp_icl", "privsv"}
    model_key = "common.model_7b_chat" if uses_chat_model else "common.model_7b"
    fallback = "meta-llama/Llama-2-7b-chat-hf" if uses_chat_model else "meta-llama/Llama-2-7b-hf"
    model = str(get_nested(settings, model_key, fallback))
    options = [("configured", "Llama-2-7b")]
    if method_id in {"cliptext", "privsv", "zo_dpga"}:
        options.append(("glm_subj", "ChatGLM3-6B"))
    selected_model = choose_from_menu("请选择模型", options)
    if selected_model is None:
        return None, None
    model = "ChatGLM3-6B" if selected_model == "glm_subj" else model

    if not choose_dataset:
        return model, None

    dataset = str(get_nested(settings, "common.dataset", "sst2"))
    datasets = [("configured", dataset)]
    if method_id in {"cliptext", "privsv", "zo_dpga"}:
        datasets.append(("subj", "Subj"))
    selected_dataset = choose_from_menu("请选择数据集", datasets)
    if selected_dataset is None:
        return None, None
    dataset = "subj" if selected_dataset == "subj" else dataset
    if (model == "ChatGLM3-6B") != (dataset.lower() == "subj"):
        print("不支持该模型/数据集组合；ChatGLM3-6B 仅支持 Subj。")
        return None, None
    return model, dataset


def _select_task(settings: dict[str, Any]) -> TaskSelection | None:
    task_kind = choose_from_menu("第一步：请选择待执行方法类型", [("finetune", "敏感信息保护微调方法"), ("evaluation", "敏感信息泄露风险多维度评估方法")])
    if task_kind is None:
        return None
    if task_kind == "finetune":
        group = choose_from_menu("第二步：请选择敏感信息保护微调类型", [(key, title) for key, (title, _) in FINETUNE_METHOD_GROUPS.items()])
        if group is None:
            return None
        group_title, method_ids = FINETUNE_METHOD_GROUPS[group]
        method_id = choose_from_menu(
            f"第三步：请选择{group_title}具体方法",
            [(key, f"{METHODS[key].title} — {METHODS[key].description}") for key in method_ids],
        )
        if method_id is None:
            return None
        # 当前/对比方法的选择发生在模型和数据集确认之后。
        return TaskSelection("finetune", method_id, "", group)

    evaluation_type = choose_from_menu("第二步：请选择敏感信息泄露风险评估方法", [("direct_attack", "直接评估"), ("indirect_attack", "间接评估")])
    if evaluation_type is None:
        return None
    finetune_type = choose_from_menu("第三步：请选择待评估模型的微调方式", [("explicit", "显式微调"), ("implicit", "隐式微调")])
    if finetune_type is None:
        return None
    groups = EVALUATION_METHOD_GROUPS[finetune_type]
    group = choose_from_menu("第四步：请选择具体微调类型", [(key, title) for key, (title, _) in groups.items()])
    if group is None:
        return None
    group_title, method_ids = groups[group]
    if group == "none":
        model = choose_from_menu(
            "请选择待评估模型",
            [("chatglm", "ChatGLM3-6b"), ("fm9g", "FM9G-7B（九格大模型）")],
        )
        if model is None:
            return None
        dataset = choose_from_menu(
            "请选择微调所用的数据集",
            [("sst2", "SST-2"), ("wikitext2", "WikiText-2")],
        )
        if dataset is None:
            return None
        explicit = finetune_type == "explicit"
        return TaskSelection(
            "evaluation", "none", evaluation_type, "none", evaluation_type, finetune_type,
            dataset=dataset,
            evaluation_model=model,
            evaluation_model_title=(
                "ChatGLM3-6b" if model == "chatglm" else "FM9G-7B（九格大模型）"
            ),
            backend_method_id="kal_dp" if explicit else "dadp_icl",
            display_method_title="显式微调" if explicit else "隐式微调",
        )
    method_id = choose_from_menu(
        f"请选择{group_title}具体方法",
        [(key, f"{METHODS[key].title} — {METHODS[key].description}") for key in method_ids],
    )
    if method_id is None:
        return None
    return TaskSelection("evaluation", method_id, evaluation_type, group, evaluation_type, finetune_type)


def _chatglm_explicit_direct_attack() -> int:
    # Kept as a small compatibility wrapper for callers that used the old
    # placeholder directly.  The interactive route below supplies settings.
    print("显式直接评估需要通过统一配置启动。")
    return 1


def _chatglm_implicit_direct_attack() -> int:
    print("隐式直接评估需要通过统一配置启动。")
    return 1


def _run_model_evaluation(selection: TaskSelection, settings: dict[str, Any]) -> int:
    """Route before resolving the legacy Kal-DP/DADP method identity."""
    from .direct_evaluation import run_direct
    from .indirect_evaluation import run_indirect
    dataset = selection.dataset or str(get_nested(settings, "common.dataset", "sst2") or "sst2")
    evaluation_label = "直接评估" if selection.experiment_id == "direct_attack" else "间接评估"
    model = "chatglm3" if selection.evaluation_model == "chatglm" else selection.evaluation_model
    finetune_label = "显式" if selection.finetune_type == "explicit" else "隐式"
    print(f"方法：{evaluation_label} 模型：{model}，微调方式：{finetune_label}，数据集：{dataset}")
    if selection.experiment_id == "direct_attack":
        try:
            mode = choose_from_menu("请选择执行模式", [("run", "立即执行"),
                                   ("history", "查看最近历史结果")])
        except (EOFError, OSError):
            # Library/test callers may intentionally invoke the route without
            # an interactive stdin; cancellation remains a successful no-op.
            return 0
        if mode is None:
            return 0
        return run_direct(
            settings, selection.evaluation_model, selection.finetune_type,
            mode=mode, dataset=dataset,
        )
    mode = choose_from_menu("请选择执行模式", [("run", "立即执行"),
                           ("history", "查看最近历史结果")])
    if mode is None:
        return 0
    return run_indirect(
        settings, selection.evaluation_model, selection.finetune_type,
        mode=mode, dataset=dataset,
    )


def interactive(settings: dict[str, Any]) -> int:
    print("\n=========================================================")
    print(" 具备敏感信息保护能力的大模型微调支撑工具集原型系统 v1.1")
    print("=========================================================")
    print("  请按提示进行后续操作 ")
    print("=========================================================")
    selection = _select_task(settings)
    if selection is None:
        print("已退出。")
        return 0
    if selection.evaluation_model in {"chatglm", "fm9g"}:
        return _run_model_evaluation(selection, settings)
    no_perturbation = selection.perturbation_group == "none"
    method_id = selection.backend_method_id or selection.method_id
    if no_perturbation:
        model, dataset = selection.evaluation_model_title, None
    else:
        model, dataset = _choose_model_and_dataset(
            settings,
            method_id=method_id,
            experiment_id=selection.experiment_id,
            choose_dataset=selection.task_kind == "finetune",
        )
    if model is None or (selection.task_kind == "finetune" and dataset is None):
        print("已退出。")
        return 0
    comparison_labels: dict[str, str] = {}
    if selection.task_kind == "finetune":
        current_method_title = METHODS[method_id].title
        baseline_finetune_type = "Prompt微调" if selection.perturbation_group == "example" else "PEFT微调"
        if model == "ChatGLM3-6B":
            baseline_finetune_type = {"cliptext": "PEFT微调", "privsv": "Prompt微调",
                                      "zo_dpga": "PEFT微调"}[method_id]
        comparison_labels = {
            # "utility": f"否，运行当前方法 {current_method_title}",
            "utility": f"否，运行当前方法",
            "baseline": f"是，运行效用对比方法 {baseline_finetune_type}",
            "performance_baseline": "是，运行性能对比方法 DP-SGD微调",
        }
        experiment_id = choose_from_menu(
            "请选择是否运行基准微调方法进行对比",
            list(comparison_labels.items()),
        )
        if experiment_id is None:
            return 0
    else:
        experiment_id = selection.experiment_id
    if model == "ChatGLM3-6B":
        from dataclasses import replace
        from .glm_subj import run as run_glm_subj
        selection = replace(selection, backend="glm_subj", model="chatglm3-6b", dataset="subj")
        mode = choose_from_menu("请选择执行模式", [("run", "立即执行"),
                               ("history", "查看最近历史结果"), ("dry", "预览命令（不执行）")])
        if mode is None:
            return 0
        # print(f"Glm-subj：{selection.method_id} / ChatGLM3-6B / Subj")
        # if method_id == "privsv":
        #     print("Private Steering 当前没有 epsilon accountant，不能宣称形式化 DP 保证。")
        return run_glm_subj(settings, selection.method_id, experiment_id, mode,
                           confirm=_confirm_real_run, selection=selection)
    method = METHODS[method_id]
    experiment = next(item for item in method.experiments if item.id == experiment_id)
    if selection.evaluation_model == "llama":
        if method_id == "dadp_icl" and experiment_id == "indirect_attack":
            _set_nested(settings, "common.model_7b_chat", get_nested(settings, "common.model_7b"))
        builder = (
            kal_no_perturbation_direct if method_id == "kal_dp" and experiment_id == "direct_attack" else
            kal_no_perturbation_indirect if method_id == "kal_dp" else
            dadp_no_perturbation_direct if experiment_id == "direct_attack" else
            dadp_no_perturbation_indirect
        )
        experiment = Experiment(experiment.id, "无扰动基准评估", experiment.description, builder=builder)
    mode = choose_from_menu("请选择执行模式", [("run", "立即执行"), ("dry", "查看最近历史结果")])
    if mode is None:
        return 0
    dry_run = mode == "dry"
    all_specs = prompt_specs(method_id, experiment_id)
    specs = [spec for spec in all_specs if spec.key not in {"common.model_7b", "common.model_7b_chat"}]
    # 其他参数设置菜单暂时关闭，全部直接采用 JSON 配置值。
    collected: list[tuple[PromptSpec, Any]] = [(spec, get_nested(settings, spec.key)) for spec in specs]
    if specs:
        print("其他实验参数：使用默认值")
    if not _prepare_runtime(method_id, experiment_id, settings, dry_run):
        return 1
    if not _prepare_assets(method_id, experiment_id, settings, all_specs, dry_run, baseline_only=no_perturbation):
        return 1
    print("\n---------------- 任务确认 ----------------")
    print(f"任务：{'微调' if selection.task_kind == 'finetune' else '评估'}")
    if not no_perturbation:
        print(f"方法：{method.title}（{method.id}）")
    if selection.task_kind == "finetune":
        comparison_confirmation = {
            "utility": "否",
            "baseline": f"是，运行 {baseline_finetune_type}",
            "performance_baseline": "是，运行 DP-SGD",
        }
        print(f"是否运行对比方法：{comparison_confirmation[experiment_id]}")
    else:
        print(f"评估类型：{'直接评估' if selection.evaluation_type == 'direct_attack' else '间接评估'}")
        print(f"微调方式：{'显式微调' if selection.finetune_type == 'explicit' else '隐式微调'}")
        finetune_group_title = "无扰动微调" if no_perturbation else FINETUNE_METHOD_GROUPS[selection.perturbation_group][0]
        print(f"微调类型：{finetune_group_title}")
    print(f"模型：{model}")
    if dataset is not None:
        print(f"数据集：{dataset}")
    print(f"模式：{'查看最近历史结果' if dry_run else '立即执行'}")
    evaluation_display_title = "Llama" if selection.task_kind == "evaluation" else None
    # for spec, value in collected:
    #     print(f"{spec.label}：{_display_value(value)}")
    python_profile = "kal_dp" if experiment_id == "performance_baseline" and method_id in {"kal_dp", "zo_dpga"} else ("dpsgd" if experiment_id == "performance_baseline" else method_id)
    check_code = doctor(
        method,
        settings,
        check_dataset=selection.task_kind == "finetune",
        python_profile=python_profile,
        display_title=(
            evaluation_display_title
            or (selection.display_method_title if no_perturbation else None)
        ),
    )
    if check_code and not dry_run:
        return check_code
    if dry_run:
        if not _yes_no("确认生成预演结果吗？"):
            return 0
    elif not _confirm_real_run():
        return 0
    return run_experiment(
        CATEGORIES[method_id], method, experiment, settings, dry_run=dry_run,
        display_title=(
            evaluation_display_title
            or (selection.display_method_title if no_perturbation else None)
        ),
        historical_baseline_only=no_perturbation,
    )


def _is_path_option(spec: PromptSpec) -> bool:
    """识别菜单中要求路径或模型标识的选项。"""
    return (
        spec.kind == "str"
        and ("路径" in spec.label or spec.key.endswith(("_path", ".model_7b", ".model_7b_chat", ".model_13b", ".mask_model")))
    )


def list_path_options(settings: dict[str, Any]) -> None:
    """遍历所有可选菜单分支，展示其中实际使用的路径配置。"""
    print("路径选项总览")
    print("  配置来源：内置默认配置，可用 --config/--set 覆盖")
    common_keys = ("results_dir", "historical_results_dir", "common.dataset_path")
    print("\n├─ 全局路径")
    for index, key in enumerate(common_keys):
        value = get_nested(settings, key, "") if "." in key else settings.get(key, "")
        print(f"│  {'└─' if index == len(common_keys) - 1 else '├─'} {key}: {value}")

    print("\n├─ ChatGLM3-6B / Subj（ClipText、PrivSV、ZO-DPGA）")
    for key in ("python.glm_subj", "glm_subj.base_model", "glm_subj.raw_data",
                "glm_subj.embedding_data", "glm_subj.output_root"):
        print(f"│  ├─ {key}: {get_nested(settings, key, '')}")
    for method in ("cliptext", "privsv", "zo_dpga"):
        for name, value in get_nested(settings, f"glm_subj.artifacts.{method}", {}).items():
            print(f"│  ├─ glm_subj.artifacts.{method}.{name}: {value or '自动解析成功训练产物'}")
    seen: set[tuple[str, str, str]] = set()
    groups = (("微调", FINETUNE_METHOD_GROUPS), ("评估", EVALUATION_METHOD_GROUPS))
    for task_title, task_groups in groups:
        print(f"\n├─ {task_title}")
        if task_title == "微调":
            branches = ((group_id, title, methods) for group_id, (title, methods) in task_groups.items())
            for group_id, group_title, method_ids in branches:
                print(f"│  ├─ {group_title}")
                for method_id in method_ids:
                    method = METHODS[method_id]
                    print(f"│  │  ├─ {method.title} ({method_id})")
                    for experiment in method.experiments:
                        specs = [spec for spec in prompt_specs(method_id, experiment.id) if _is_path_option(spec)]
                        if not specs:
                            continue
                        print(f"│  │  │  └─ {experiment.title} ({experiment.id})")
                        for spec in specs:
                            key = (method_id, experiment.id, spec.key)
                            if key in seen:
                                continue
                            seen.add(key)
                            print(f"│  │  │     ├─ {spec.label} [{spec.key}]: {get_nested(settings, spec.key, '<未配置>')}")
        else:
            for evaluation_type, evaluation_groups in task_groups.items():
                evaluation_title = "直接评估" if evaluation_type == "explicit" else "间接评估"
                print(f"│  ├─ {evaluation_title}")
                for group_id, (group_title, method_ids) in evaluation_groups.items():
                    print(f"│  │  ├─ {group_title}")
                    for method_id in method_ids:
                        if method_id == "none":
                            print("│  │  │  └─ 无扰动（使用对应评估接口配置）")
                            continue
                        method = METHODS[method_id]
                        print(f"│  │  │  ├─ {method.title} ({method_id})")
                        experiment_id = "direct_attack" if evaluation_type == "explicit" else "indirect_attack"
                        specs = [spec for spec in prompt_specs(method_id, experiment_id) if _is_path_option(spec)]
                        for spec in specs:
                            key = (method_id, experiment_id, spec.key)
                            if key in seen:
                                continue
                            seen.add(key)
                            print(f"│  │  │     ├─ {spec.label} [{spec.key}]: {get_nested(settings, spec.key, '<未配置>')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="具备敏感信息保护能力的大模型微调支撑工具集")
    parser.add_argument("--config", help="JSON 配置文件；不填则使用内置默认配置")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置，可重复")
    subparsers = parser.add_subparsers(dest="action")

    list_parser = subparsers.add_parser("list", help="列出六种方法及其全部实验")
    list_parser.add_argument("--method", choices=METHODS.keys())

    subparsers.add_parser(
        "paths",
        help="遍历全部菜单分支，按层次列出会使用的路径选项",
    )

    doctor_parser = subparsers.add_parser("doctor", help="检查代码、Python、模型和数据集")
    doctor_parser.add_argument("--method", choices=METHODS.keys())

    run_parser = subparsers.add_parser("run", help="跳过菜单，直接执行指定实验")
    run_parser.add_argument("--method", required=True, choices=METHODS.keys())
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--backend", choices=["legacy", "glm_subj"], default="legacy")
    run_parser.add_argument("--history", action="store_true", help="读取所选后端的历史结果")
    run_parser.add_argument("--dry-run", action="store_true", help="只生成命令，不执行实验")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        settings = load_settings(args.config, args.set)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.action is None:
        return interactive(settings)
    if args.action == "list":
        list_catalog(METHODS, args.method)
        return 0
    if args.action == "paths":
        list_path_options(settings)
        return 0
    if args.action == "doctor":
        selected = [METHODS[args.method]] if args.method else list(METHODS.values())
        return max((doctor(method, settings) for method in selected), default=0)
    if args.action == "run":
        if args.backend == "glm_subj":
            from .glm_subj import run as run_glm_subj
            if args.method not in {"cliptext", "privsv", "zo_dpga"}:
                parser.error("Glm-subj 仅支持 ClipText、PrivSV、ZO-DPGA")
            return run_glm_subj(settings, args.method, args.experiment,
                               "history" if args.history else "dry" if args.dry_run else "run")
        if args.history:
            parser.error("--history 用于 Glm-subj 后端")
        method = METHODS[args.method]
        try:
            experiment = next(item for item in method.experiments if item.id == args.experiment)
        except StopIteration:
            available = ", ".join(item.id for item in method.experiments)
            parser.error(f"方法 {method.id} 不包含实验 {args.experiment}；可选：{available}")
        return run_experiment(CATEGORIES[method.id], method, experiment, settings, dry_run=args.dry_run)
    parser.error("未知操作")
    return 2
