from __future__ import annotations

import argparse
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from privacy_tools.bootstrap import check_imports, environment_plan
from privacy_tools.catalog import CLIP_SST, PRIVSV_ROOT
from privacy_tools.core import (
    _materialize_patched_step,
    evaluation_banner,
    load_settings,
    run_experiment,
    write_metrics_result,
)
from privacy_tools.unified import CATEGORIES, METHOD_GROUPS, METHODS, _confirm_real_run, prompt_specs


EXPECTED_CHOICES = 24
EXPECTED_KINDS = {"experiment": 24}
EXPECTED_EXPERIMENTS = (
    ("utility", "本微调方法"),
    ("baseline", "对比微调方法"),
    ("direct_attack", "直接评估"),
    ("indirect_attack", "间接评估"),
)
EXPECTED_GROUPS = {
    "example": ("示例扰动", ("dadp_icl", "privsv")),
    "embedding": ("嵌入扰动", ("cliptext", "ptcds")),
    "gradient": ("梯度扰动", ("kal_dp", "zo_dpga")),
}
EXPECTED_PATH_NAMES = {
    "cliptext": "ClipText",
    "ptcds": "PTCDS",
    "kal_dp": "Kal-DP",
    "zo_dpga": "ZO-DPGA",
    "dadp_icl": "DADP-ICL",
    "privsv": "PrivSV",
}
FORBIDDEN_PUBLIC_NAMES = (
    "".join(("San", "Text")),
    "".join(("DP", "Zero")),
    "".join(("DP", "ICL")),
)

LIGHT_EXECUTIONS: dict[str, tuple[str, ...]] = {}


def _semantic_failures(label: str, actual, source_text: str) -> list[str]:
    """Check integration contracts that syntax compilation alone cannot detect."""
    failures: list[str] = []
    normalized_source = source_text.replace("\\\\", "\\")
    script_name = Path(actual.command[1]).name
    if (
        label.endswith("/direct_attack")
        and label.split("/", 1)[0] != "privsv"
        and script_name == "direct_attack_runner.py"
    ):
        if source_text.count("PeftModel.from_pretrained") != 1:
            failures.append("通用攻击应且只能加载一个受保护 adapter")
        if "requested_samples" not in source_text or "generated_samples" not in source_text:
            failures.append("通用攻击未记录请求/实生成样本数")
    if label == "ptcds/direct_attack":
        is_protected = script_name.endswith("extraction_temperature_decay.py") and not script_name.endswith("extraction_temperature_decay_base.py")
        is_baseline = script_name.endswith("extraction_temperature_decay_base.py")
        is_member = script_name.endswith("member_test.py")
        if not (is_protected or is_baseline or is_member):
            failures.append("PTCDS 直接评估混入两路前缀抽取流程以外的脚本")
        if is_protected:
            for option in ("--model_name", "--baseline_adapter", "--protected_adapter", "--outfile"):
                if option not in actual.command:
                    failures.append(f"PTCDS 受保护模型抽取缺少 {option}")
            if "--baseline_adapter" in actual.command and "--protected_adapter" in actual.command:
                baseline_value = actual.command[actual.command.index("--baseline_adapter") + 1]
                protected_value = actual.command[actual.command.index("--protected_adapter") + 1]
                if baseline_value != protected_value:
                    failures.append("PTCDS 受保护分支没有像 ClipText 一样只加载受保护权重")
        if is_baseline:
            if "--outfile" not in actual.command:
                failures.append("PTCDS base 分支未指定独立输出文件")
            if "../outputs/llama2-sst2-lora-base/base-checkpoint-6500" in source_text:
                failures.append("PTCDS base 分支未绑定配置中的未保护权重")
        if is_member and "protected_extraction_temperature_decay.txt" not in source_text and "baseline_extraction_temperature_decay.txt" not in source_text:
            failures.append("PTCDS member_test 未绑定本次受保护或 base 抽取结果")
    if label in {"privsv/utility", "privsv/baseline"}:
        if str(PRIVSV_ROOT / "datasets" / "generate") not in normalized_source:
            failures.append("PrivSV 训练数据仍依赖当前工作目录")
        if str(PRIVSV_ROOT / "datasets" / "test") not in normalized_source:
            failures.append("PrivSV 测试数据仍依赖当前工作目录")
        if "clip=args.clip" not in source_text:
            failures.append("PrivSV 的 --clip 参数未传入实验函数")
    if label == "privsv/direct_attack" and "extract_topn-sv" in script_name:
        python_path = actual.env.get("PYTHONPATH", "")
        if str(PRIVSV_ROOT) not in python_path:
            failures.append("PrivSV 攻击脚本无法导入父目录 steering_wrapper")
        checkpoint = PRIVSV_ROOT / "private_steering_vectors_llama_7b_hf.pt"
        if str(checkpoint) not in normalized_source:
            failures.append("PrivSV 攻击脚本未绑定私有 steering checkpoint")
    if label.endswith("/indirect_attack"):
        method_id = label.split("/", 1)[0]
        if method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"}:
            if script_name.endswith("neighbour_attack.py") or script_name.endswith("neighbour_attack_wo.py"):
                for option in ("--proc-id", "--model", "--dataset", "--split", "--attack-gpu", "--search-gpu"):
                    if option not in actual.command:
                        failures.append(f"原生 adapter 邻居攻击缺少 {option}")
                if "../../models/Llama-2-7b-hf" in source_text:
                    failures.append("原生 adapter 邻居攻击仍使用失效的相对基础模型路径")
                if method_id == "cliptext" and "../../Kal-DP/" in source_text:
                    failures.append("ClipText 原生邻居攻击仍错误绑定 Kal-DP adapter")
            elif script_name != "evaluate_mia_fair_compare.py":
                failures.append("原生 adapter 邻居攻击混入非原生生成器")
        elif method_id in {"dadp_icl", "privsv"}:
            expected_script = "neighbour_mia.py" if method_id == "dadp_icl" else "neighbour_mia_sv.py"
            if script_name.endswith(expected_script):
                for option in ("--proc-id", "--model", "--dataset"):
                    if option not in actual.command:
                        failures.append(f"原生示例邻居攻击缺少 {option}")
                if "load_dataset(\"glue\", \"sst2\")" in source_text:
                    failures.append("原生示例邻居攻击仍依赖在线 SST-2 数据下载")
            elif script_name.endswith("combine.py"):
                protected = "本方法" in actual.name
                expected_label = "本方法" if protected else "对比方法"
                expected_file = "attack_results_bert_combined_noise.csv" if protected else "attack_results_bert_combined.csv"
                print_lines = [line for line in source_text.splitlines() if "已合并成员+非成员文件" in line]
                if len(print_lines) != 1:
                    failures.append("间接评估 combine 应且只能输出一条实际保存位置")
                else:
                    output_line = print_lines[0]
                    if f"当前评估：{expected_label}" not in output_line:
                        failures.append(f"间接评估 combine 未输出当前评估对象 {expected_label}")
                    if expected_file not in output_line:
                        failures.append(f"间接评估 combine 输出文件名未同步为 {expected_file}")
                    if not protected and "attack_results_bert_combined_noise.csv" in output_line:
                        failures.append("对比方法第二次 combine 仍错误输出 noise 文件名")
            elif script_name.endswith("evaluate_attack_metrics.py"):
                pass
            else:
                failures.append("示例方法间接攻击未调用自身原生邻居 MIA 脚本")
    if label == "dadp_icl/direct_attack" and script_name == "invade_prefix_with_custom.py":
        for option in ("--use-custom-prefixes", "--custom-prefix-file", "--N", "--outfile"):
            if option not in actual.command:
                failures.append(f"DADP-ICL 实际直接攻击缺少 {option}")
    return failures


def structural_audit() -> int:
    settings = load_settings(None, [])
    failures: list[str] = []
    if settings["methods"]["cliptext"].get("epsilon") != 10.0:
        failures.append("ClipText 默认隐私预算必须为 epsilon=10.0")
    if settings["methods"]["kal_dp"].get("dpsgd_max_physical_batch_size") != 12:
        failures.append("Kal-DP 的 DP-SGD 默认 max_physical_batch_size 必须为 12")
    confirmation_output = io.StringIO()
    with patch("builtins.input", side_effect=["输错", "RUN"]), redirect_stdout(confirmation_output):
        if not _confirm_real_run():
            failures.append("真实运行确认未在输错后重新接受输入")
    if "请重新输入小写 run" not in confirmation_output.getvalue():
        failures.append("真实运行确认输错后没有给出重试提示")
    with patch("builtins.input", return_value="0"), redirect_stdout(io.StringIO()):
        if _confirm_real_run():
            failures.append("真实运行确认输入 0 后没有取消")
    count = 0
    kind_counts = {key: 0 for key in EXPECTED_KINDS}
    if METHOD_GROUPS != EXPECTED_GROUPS:
        failures.append(f"入口类别/方法顺序应为 {EXPECTED_GROUPS}，实际为 {METHOD_GROUPS}")
    with tempfile.TemporaryDirectory(prefix="privacy-tool-audit-") as temporary:
        temporary_root = Path(temporary)
        for method_id, method in METHODS.items():
            seen: set[str] = set()
            if method.path_name != EXPECTED_PATH_NAMES[method_id]:
                failures.append(f"{method_id}: 交付路径名应为 {EXPECTED_PATH_NAMES[method_id]}，实际为 {method.path_name}")
            if method.root.name != EXPECTED_PATH_NAMES[method_id]:
                failures.append(f"{method_id}: 源码顶层目录名不规范：{method.root}")
            public_text = " ".join(
                [method.title, method.description, *method.warnings]
                + [part for item in method.experiments for part in (item.title, item.description)]
            )
            for forbidden in FORBIDDEN_PUBLIC_NAMES:
                if forbidden.casefold() in public_text.casefold():
                    failures.append(f"{method_id}: 对外文案仍出现旧名称 {forbidden}")
            if method_id not in CATEGORIES:
                failures.append(f"{method_id}: 缺少结果类别")
            actual_experiments = tuple((item.id, item.title) for item in method.experiments)
            if actual_experiments != EXPECTED_EXPERIMENTS:
                failures.append(f"{method_id}: 四项菜单不符合统一结构：{actual_experiments}")
            for experiment in method.experiments:
                count += 1
                if experiment.kind not in kind_counts:
                    failures.append(f"{method_id}/{experiment.id}: 未知运行项类型 {experiment.kind}")
                else:
                    kind_counts[experiment.kind] += 1
                label = f"{method_id}/{experiment.id}"
                if experiment.id in seen:
                    failures.append(f"{label}: 实验 ID 重复")
                seen.add(experiment.id)

                plan = environment_plan(method_id, experiment.id)
                if plan.requirements is not None and not plan.requirements.is_file():
                    failures.append(f"{label}: 缺少依赖文件 {plan.requirements}")
                specs = prompt_specs(method_id, experiment.id)
                spec_keys = [item.key for item in specs]
                if len(spec_keys) != len(set(spec_keys)):
                    failures.append(f"{label}: 交互参数重复")

                run_dir = temporary_root / method_id / experiment.id
                run_dir.mkdir(parents=True, exist_ok=True)
                if experiment.internal_report:
                    print(f"[ OK ] {label:<36} 内部结果汇总")
                    continue
                try:
                    steps = experiment.builder(settings, run_dir)
                except Exception as exc:
                    failures.append(f"{label}: 构造命令失败：{exc}")
                    continue
                if not steps:
                    failures.append(f"{label}: 没有执行步骤")
                    continue
                if experiment.id in {"direct_attack", "indirect_attack"}:
                    for step in steps:
                        banner = evaluation_banner(method, experiment, step)
                        if not banner or method.title not in banner:
                            failures.append(f"{label}: 评估步骤未明确输出当前方法：{step.name}")
                        elif "当前方法评估流程" in banner:
                            failures.append(f"{label}: 评估步骤未区分保护/非保护状态：{step.name}")
                if experiment.id == "indirect_attack":
                    expected_steps = 5 if method_id in {"cliptext", "ptcds", "kal_dp", "zo_dpga"} else 8
                    if len(steps) != expected_steps:
                        failures.append(f"{label}: 原生/补充间接攻击应有 {expected_steps} 个步骤，实际为 {len(steps)}")
                if label == "ptcds/direct_attack" and len(steps) != 4:
                    failures.append(f"{label}: 应与 ClipText 一样包含两路抽取和两路 member_test，共 4 步，实际为 {len(steps)}")
                for index, step in enumerate(steps, 1):
                    if label in {"ptcds/utility", "ptcds/baseline"}:
                        command = step.command
                        if Path(command[1]).name != "finetune_llama2_sst2_lora_data.py":
                            failures.append(f"{label}: 未按 PTCDS/md.md 调用 LoRA 微调脚本")
                        required = {
                            "--eval_steps": "948",
                            "--wandb_project": "llama2-sst2-lora",
                            "--run_name": "llama2-7b-sst2",
                        }
                        for option, expected in required.items():
                            if option not in command or command[command.index(option) + 1] != expected:
                                failures.append(f"{label}: {option} 未严格采用 PTCDS/md.md 的 {expected}")
                        forbidden = {"--target_layer", "--cluster_num", "--recluster", "--eps"}
                        if any(option in command for option in forbidden):
                            failures.append(f"{label}: 仍混入旧版隐层聚类训练参数")
                    if label == "kal_dp/baseline":
                        command = step.command
                        expected_options = {
                            "--batch_size": "32",
                            "--max_physical_batch_size": "12",
                            "--seq_length": "128",
                            "--is_lora": "True",
                            "--lora_rank": "16",
                            "--learning_rate": "0.0001",
                            "--max_grad_norm": "3.0",
                            "--epochs": "2",
                        }
                        for option, expected in expected_options.items():
                            if option not in command or command[command.index(option) + 1] != expected:
                                failures.append(f"{label}: {option} 未严格采用 Kal-DP/readme.md 的 {expected}")
                    if label in {"dadp_icl/utility", "dadp_icl/baseline"} and Path(step.command[1]).name != "run_exp_sst2_loca_monitor.py":
                        failures.append(f"{label}: 未使用原始代码中带时间、吞吐率和内存监测的实验脚本")
                    if not step.cwd.exists():
                        failures.append(f"{label} step {index}: cwd 不存在：{step.cwd}")
                    if len(step.command) < 2:
                        failures.append(f"{label} step {index}: 命令不完整")
                        continue
                    source = step.source_script or Path(step.command[1])
                    if not source.is_file():
                        failures.append(f"{label} step {index}: 脚本不存在：{source}")
                        continue
                    try:
                        actual = _materialize_patched_step(step, run_dir, index)
                        actual_source = Path(actual.command[1])
                        text = actual_source.read_text(encoding="utf-8")
                        compile(text, str(actual_source), "exec")
                        for issue in _semantic_failures(label, actual, text):
                            failures.append(f"{label} step {index}: {issue}")
                    except Exception as exc:
                        failures.append(f"{label} step {index}: 运行时适配或语法失败：{exc}")
                print(f"[ OK ] {label:<36} {len(steps)} 个步骤")

        # Exercise a single-sample attack edge case.
        variant = load_settings(None, [])
        for method_id in ("cliptext", "ptcds", "kal_dp", "zo_dpga"):
            variant["methods"][method_id]["attack_samples"] = 1
            experiment = next(
                item for item in METHODS[method_id].experiments if item.id == "direct_attack"
            )
            step = experiment.builder(variant, temporary_root / f"edge-{method_id}-attack")[0]
            batch_option = "--batch_size"
            batch_index = step.command.index(batch_option) + 1
            if step.command[batch_index] != "1":
                failures.append(f"{method_id}/direct_attack: 单样本攻击仍使用大于 1 的 batch")

        # Verify the single CSV metric result and unit conversions without loading a model.
        metric_run = temporary_root / "metrics" / "utility" / "20260804_000000_test"
        metric_run.mkdir(parents=True)
        (metric_run / "stdout.log").write_text(
            "Accuracy: 0.9123\nPeak CUDA Memory: 2048 MB\nOverall runtime (s): 3600\nThroughput: 12.5 samples/s\n",
            encoding="utf-8",
        )
        metric_experiment = next(item for item in METHODS["kal_dp"].experiments if item.id == "utility")
        metrics_path = write_metrics_result(
            metric_run, METHODS["kal_dp"], metric_experiment, 0, 0.0, [], settings
        )
        metric_text = metrics_path.read_text(encoding="utf-8-sig")
        for expected in ("accuracy_pct", "peak_memory_gb", "runtime_hours", "throughput_samples_per_second", "91.23", "2.0", "1.0", "12.5"):
            if expected not in metric_text:
                failures.append(f"统一指标 CSV 缺少预期字段或换算值：{expected}")
        if (metric_run / "result.json").exists() or (metric_run / "report.md").exists():
            failures.append("指标汇总仍生成 result.json 或 report.md")

        privsv_run = temporary_root / "metrics" / "privsv" / "20260804_000001_test"
        privsv_run.mkdir(parents=True)
        privsv_log = privsv_run / "experiment_logs.jsonl"
        privsv_log.write_text(
            '{"steering_type":"private_mean","multiplier":2,"result":0.7926,"stage":"eval",'
            '"metrics":{"elapsed_seconds":60,"throughput_samples_per_second":8.5,'
            '"peak_gpu_memory_allocated_mb":1024}}\n',
            encoding="utf-8",
        )
        privsv_experiment = next(item for item in METHODS["privsv"].experiments if item.id == "utility")
        privsv_metrics = write_metrics_result(
            privsv_run, METHODS["privsv"], privsv_experiment, 0, 0.0, [privsv_log], settings
        ).read_text(encoding="utf-8-sig")
        for expected in ("79.26", "1.0", "8.5"):
            if expected not in privsv_metrics:
                failures.append(f"PrivSV 指标映射缺少预期值：{expected}")

        zo_run = temporary_root / "metrics" / "zo" / "20260804_000002_test"
        zo_run.mkdir(parents=True)
        zo_benchmark = zo_run / "benchmark.json"
        zo_benchmark.write_text(
            '{"privacy":{"epsilon":4},"training_wall_seconds":1800,'
            '"gpu_stats":[{"peak_allocated_bytes":2147483648,"peak_reserved_bytes":1073741824}],'
            '"trainer_metrics":{"train_samples_per_second":27.7}}',
            encoding="utf-8",
        )
        zo_evaluation = zo_run / "evaluation_benchmark.json"
        zo_evaluation.write_text(
            '[{"metrics":{"accuracy":0.8581},"examples_per_second":30}]',
            encoding="utf-8",
        )
        zo_experiment = next(item for item in METHODS["zo_dpga"].experiments if item.id == "utility")
        zo_metrics = write_metrics_result(
            zo_run, METHODS["zo_dpga"], zo_experiment, 0, 0.0,
            [zo_benchmark, zo_evaluation], settings
        ).read_text(encoding="utf-8-sig")
        for expected in ("4.0", "85.81", "2.0", "0.5", "27.7"):
            if expected not in zo_metrics:
                failures.append(f"ZO-DPGA 指标映射缺少预期值：{expected}")

    if count != EXPECTED_CHOICES:
        failures.append(f"运行项总数应为 {EXPECTED_CHOICES}，实际为 {count}")
    if kind_counts != EXPECTED_KINDS:
        failures.append(f"运行项分类应为 {EXPECTED_KINDS}，实际为 {kind_counts}")
    attack_roots = [
        METHODS["cliptext"].root / "clip-attack",
        METHODS["ptcds"].root / "attack",
        METHODS["kal_dp"].root / "attack",
        METHODS["dadp_icl"].root / "direct_attack",
        METHODS["dadp_icl"].root / "Indirect_attack",
        METHODS["privsv"].root / "direct_attack",
        METHODS["privsv"].root / "Indirect_attack",
    ]
    for attack_root in attack_roots:
        for source in attack_root.rglob("*"):
            if source.is_file() and source.suffix.lower() in {".py", ".md", ".json", ".sh"}:
                text = source.read_text(encoding="utf-8-sig", errors="replace")
                if any(prefix in text for prefix in ("/home/", "//home/", "/root/", "/mnt/")):
                    failures.append(f"攻击源码仍含服务器绝对路径：{source}")
    print(f"\n结构审计：{len(METHODS)} 种方法，{count} 个运行项：{kind_counts}。")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        print(f"结构审计失败：{len(failures)} 项。")
        return 1
    print("结构审计通过：全部运行项均能构造命令，所需脚本、环境方案和运行时适配语法正常。")
    return 0


def execute_lightweight() -> int:
    settings = load_settings(None, [])
    for method_id in METHODS:
        settings.setdefault("python", {})[method_id] = sys.executable
    failures: list[str] = []
    executed = 0
    for method_id, experiment_ids in LIGHT_EXECUTIONS.items():
        method = METHODS[method_id]
        for experiment_id in experiment_ids:
            experiment = next(item for item in method.experiments if item.id == experiment_id)
            plan = environment_plan(method_id, experiment_id)
            ready, missing = check_imports(sys.executable, plan.imports)
            label = f"{method_id}/{experiment_id}"
            if not ready:
                failures.append(f"{label}: 当前 Python 缺少 {', '.join(missing)}")
                continue
            print(f"\n[EXEC] {label}")
            try:
                code = run_experiment(CATEGORIES[method_id], method, experiment, settings, dry_run=False)
            except Exception as exc:
                failures.append(f"{label}: 未捕获异常：{exc}")
                continue
            executed += 1
            if code != 0:
                failures.append(f"{label}: return code={code}")

    print(f"\n轻量实跑：共执行 {executed} 个实验。")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        print(f"轻量实跑失败：{len(failures)} 项。")
        return 1
    print("轻量实跑通过。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="穷举检查六种方法的全部运行项")
    parser.add_argument("--execute-light", action="store_true", help="额外真实执行不加载大模型的轻量实验")
    args = parser.parse_args()
    code = structural_audit()
    if code != 0 or not args.execute_light:
        return code
    return execute_lightweight()


if __name__ == "__main__":
    raise SystemExit(main())
