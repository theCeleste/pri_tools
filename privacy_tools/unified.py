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
    create_environment,
    download_glove,
    environment_plan,
    environment_python,
    ensure_nltk_stopwords,
)
from .catalog import embedding_methods, example_methods, gradient_methods
from .core import (
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
    "example": ("示例扰动", ("dadp_icl", "privsv")),
    "embedding": ("嵌入扰动", ("cliptext", "ptcds")),
    "gradient": ("梯度扰动", ("kal_dp", "zo_dpga")),
}


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    kind: str = "str"
    hint: str = ""


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
            PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 私有 steering checkpoint 路径", "str"),
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
        if experiment_id == "baseline":
            return [MODEL, GPU, CLIP_BATCH, CLIP_EPOCHS]
        return []
    if method_id == "ptcds":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id)
        if experiment_id == "indirect_attack":
            return _native_adapter_indirect_specs(method_id)
        if experiment_id == "baseline":
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
        if experiment_id == "baseline":
            return [MODEL, GPU, DADP_VALID, DADP_TEST]
        return [MODEL, GPU, DADP_SIGMA, DADP_TRAIN, DADP_SPLITS, DADP_VALID, DADP_TEST, DADP_TOKENS]
    if method_id == "privsv":
        if experiment_id == "direct_attack":
            return _attack_specs(method_id, adapter=False) + [
                PromptSpec("methods.privsv.baseline_adapter_path", "未保护对比 adapter 路径或 Hugging Face 名称", "str"),
                PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 私有 steering checkpoint 路径", "str")
            ]
        if experiment_id == "indirect_attack":
            return [
                MODEL_CHAT,
                GPU,
                PromptSpec("methods.privsv.steering_checkpoint_path", "PrivSV 私有 steering checkpoint 路径", "str"),
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
        confirm = input("真实实验可能耗时较长。请输入 run 确认；输入 0 取消：").strip().lower()
        if confirm == "run":
            return True
        if confirm in {"0", "q", "quit", "cancel", "取消"}:
            return False
        print("输入不正确，请重新输入小写 run；如需取消请输入 0。")


def _prepare_runtime(method_id: str, experiment_id: str, settings: dict[str, Any], dry_run: bool) -> bool:
    plan = environment_plan(method_id, experiment_id)
    if dry_run:
        _set_nested(settings, f"python.{method_id}", sys.executable)
        print(f"  安全预演使用当前 Python：{sys.executable}")
        print("  安全预演不会安装依赖。")
        return True
    if plan.level == "none":
        _set_nested(settings, f"python.{method_id}", sys.executable)
        print(f"  {plan.explanation}")
        print(f"  使用当前 Python：{sys.executable}")
        return True

    print(f"  {plan.explanation}")
    strategy = choose_from_menu(
        "请选择 Python 环境准备方式",
        [
            ("managed", "自动创建隔离环境并安装缺失依赖（推荐）"),
            ("current", f"使用当前环境：{sys.executable}"),
            ("existing", "输入一个已经准备好的 Python 路径"),
        ],
    )
    if strategy is None:
        return False
    if strategy == "managed":
        selected = environment_python(method_id, plan.level)
    elif strategy == "current":
        selected = Path(sys.executable)
    else:
        raw = input("请输入 Python 解释器完整路径：").strip()
        if not raw:
            print("没有输入路径，已取消。")
            return False
        selected = Path(raw).expanduser()

    ready, missing = check_imports(selected, plan.imports)
    if ready:
        print(f"  [OK] 所需依赖已就绪：{selected}")
        _set_nested(settings, f"python.{method_id}", str(selected))
        return True

    print(f"  [缺少依赖] {', '.join(missing)}")
    if strategy != "managed":
        if not _yes_no("是否改用自动隔离环境？"):
            print("已停止。当前环境缺少实验依赖。")
            return False
        selected = environment_python(method_id, plan.level)
        ready, missing = check_imports(selected, plan.imports)
        if ready:
            _set_nested(settings, f"python.{method_id}", str(selected))
            return True

    print(f"  将准备：{selected}")
    print(f"  依赖文件：{plan.requirements}")
    print("  需要网络连接；训练环境通常需要数 GB 磁盘空间。")
    if not _yes_no("现在自动创建并安装吗？"):
        print("已取消环境安装。")
        return False
    code, selected = create_environment(method_id, plan)
    if code != 0:
        print(f"环境安装失败（return code={code}）。上方 pip/conda 输出包含具体原因。")
        return False
    ready, missing = check_imports(selected, plan.imports)
    if not ready:
        print(f"环境仍缺少依赖：{', '.join(missing)}")
        return False
    print(f"  [OK] 隔离环境准备完成：{selected}")
    _set_nested(settings, f"python.{method_id}", str(selected))
    return True


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~", "\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _prepare_assets(method_id: str, experiment_id: str, settings: dict[str, Any], specs: list[PromptSpec], dry_run: bool) -> bool:
    if dry_run:
        print("  安全预演只检查命令结构，不下载模型或数据。")
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
        elif model.startswith("meta-llama/") and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            print("\n[模型权限提示] Llama-2 通常需要先在 Hugging Face 接受许可。")
            print("如果模型已完整缓存在服务器，可不输入 token。token 只用于本次进程，不会写入结果文件。")
            if _yes_no("现在输入 Hugging Face token 吗？", default=False):
                token = getpass.getpass("HF token（输入不回显）：").strip()
                if token:
                    os.environ["HF_TOKEN"] = token

    if experiment_id == "direct_attack" and method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"}:
        adapter_keys = [f"methods.{method_id}.adapter_path", f"methods.{method_id}.baseline_adapter_path"]
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
        adapter_keys = [f"methods.{method_id}.adapter_path"]
        if method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"}:
            adapter_keys.append(f"methods.{method_id}.baseline_adapter_path")
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
                print(f"  [OK] PrivSV 私有 steering checkpoint：{resolved}")
                _set_nested(settings, key, str(resolved))
                break
            print(f"\n[缺少 PrivSV 私有 steering checkpoint] 当前文件不存在：{resolved}")
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


def interactive(settings: dict[str, Any]) -> int:
    print("\n================================================")
    print("  具备敏感信息保护能力的大模型微调支撑工具集")
    print("  按提示选择即可；工具集会自动检查环境、依赖、模型和数据")
    print("  方括号内是默认值，直接回车即可采用")
    print("================================================")
    category = choose_from_menu(
        "第一步：请选择方法类别",
        [(key, title) for key, (title, _) in METHOD_GROUPS.items()],
    )
    if category is None:
        print("已退出。")
        return 0
    category_title, method_ids = METHOD_GROUPS[category]
    method_id = choose_from_menu(
        f"第二步：请选择{category_title}方法",
        [(key, f"{METHODS[key].title} — {METHODS[key].description}") for key in method_ids],
    )
    if method_id is None:
        print("已退出。")
        return 0
    method = METHODS[method_id]
    experiment_id = choose_from_menu(
        "第三步：请选择运行项",
        [(item.id, f"{item.title} — {item.description}") for item in method.experiments],
    )
    if experiment_id is None:
        print("已退出。")
        return 0
    experiment = next(item for item in method.experiments if item.id == experiment_id)

    mode = choose_from_menu(
        "第四步：请选择执行模式",
        [
            ("dry", "安全预演（推荐）：生成完整命令，不启动原实验"),
            ("run", "真实执行：启动原实验，可能占用 GPU 较长时间"),
        ],
    )
    if mode is None:
        print("已退出。")
        return 0
    dry_run = mode == "dry"

    print("\n第五步：请选择参数设置方式")
    collected: list[tuple[PromptSpec, Any]] = []
    specs = prompt_specs(method_id, experiment_id)
    if specs:
        parameter_mode = choose_from_menu(
            "参数设置",
            [
                ("defaults", "一键使用全部默认参数（推荐）"),
                ("custom", "逐项输入并调整参数"),
            ],
        )
        if parameter_mode is None:
            print("已退出。")
            return 0
        if parameter_mode == "defaults":
            print("  已采用该实验的全部默认参数。")
            collected = [(spec, get_nested(settings, spec.key)) for spec in specs]
        else:
            print("  逐项输入；直接回车可保留该项默认值：")
            for spec in specs:
                value = _ask_setting(settings, spec)
                _set_nested(settings, spec.key, value)
                collected.append((spec, value))
    else:
        print("  该运行项不需要额外参数，继续使用内置设置。")

    print("\n第六步：自动准备与检查")
    if not _prepare_runtime(method_id, experiment_id, settings, dry_run):
        return 1
    if not _prepare_assets(method_id, experiment_id, settings, specs, dry_run):
        print("前置资源尚未准备好，本次未启动实验。")
        return 1

    print("\n---------------- 第七步：任务确认 ----------------")
    print(f"类别：{category_title}")
    print(f"方法：{method.title}（{method.id}）")
    print(f"实验：{experiment.title}（{experiment.id}）")
    print(f"模式：{'安全预演 dry-run' if dry_run else '真实执行'}")
    print(f"Python：{get_nested(settings, f'python.{method_id}')}")
    print(f"结果根目录：{project_path(str(settings.get('results_dir', 'privacy_tools/results')))}")
    for spec, value in collected:
        print(f"{spec.label}：{_display_value(value)}")
    print("--------------------------------------------------")

    check_code = doctor(method, settings)
    if check_code and not dry_run:
        print("\n环境检查存在阻塞，已停止真实执行。修复上面的 [FAIL] 后重试。")
        return check_code
    if dry_run:
        if not _yes_no("确认生成预演结果吗？"):
            print("已取消。")
            return 0
    else:
        if not _confirm_real_run():
            print("已取消。")
            return 0
    return run_experiment(CATEGORIES[method_id], method, experiment, settings, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="具备敏感信息保护能力的大模型微调支撑工具集")
    parser.add_argument("--config", help="JSON 配置文件；不填则使用内置默认配置")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置，可重复")
    subparsers = parser.add_subparsers(dest="action")

    list_parser = subparsers.add_parser("list", help="列出六种方法及其全部实验")
    list_parser.add_argument("--method", choices=METHODS.keys())

    doctor_parser = subparsers.add_parser("doctor", help="检查代码、Python 和模型")
    doctor_parser.add_argument("--method", choices=METHODS.keys())

    run_parser = subparsers.add_parser("run", help="跳过菜单，直接执行指定实验")
    run_parser.add_argument("--method", required=True, choices=METHODS.keys())
    run_parser.add_argument("--experiment", required=True)
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
    if args.action == "doctor":
        selected = [METHODS[args.method]] if args.method else list(METHODS.values())
        return max((doctor(method, settings) for method in selected), default=0)
    if args.action == "run":
        method = METHODS[args.method]
        try:
            experiment = next(item for item in method.experiments if item.id == args.experiment)
        except StopIteration:
            available = ", ".join(item.id for item in method.experiments)
            parser.error(f"方法 {method.id} 不包含实验 {args.experiment}；可选：{available}")
        return run_experiment(CATEGORIES[method.id], method, experiment, settings, dry_run=args.dry_run)
    parser.error("未知操作")
    return 2
