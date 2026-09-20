from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_PATHS = [
    ROOT / "privacy_tool.py",
    ROOT / "full_audit.py",
    ROOT / "privacy_tools" / "core.py",
    ROOT / "privacy_tools" / "catalog.py",
    ROOT / "privacy_tools" / "unified.py",
    ROOT / "privacy_tools" / "bootstrap.py",
    ROOT / "privacy_tools" / "direct_attack_runner.py",
    ROOT / "privacy_tools" / "prompt_direct_attack_runner.py",
    ROOT / "privacy_tools" / "indirect_attack_runner.py",
    ROOT / "privacy_tools" / "native_csv_attack_evaluator.py",
    ROOT / "privacy_tools" / "requirements" / "auto" / "analysis.txt",
    ROOT / "privacy_tools" / "datasets" / "sst2" / "train.tsv",
    ROOT / "privacy_tools" / "reference_assets" / "shared_legacy_attack" / "extraction_temperature_decay.txt",
    ROOT / "ClipText",
    ROOT / "PTCDS",
    ROOT / "PTCDS" / "attack" / "extraction_temperature_decay.py",
    ROOT / "PTCDS" / "attack" / "all_scores_sst2_roberta_0_train.pkl",
    ROOT / "Kal-DP",
    ROOT / "ZO-DPGA",
    ROOT / "DADP-ICL",
    ROOT / "PrivSV",
]


def run(label: str, command: list[str]) -> bool:
    print(f"\n[TEST] {label}")
    print("       " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 0:
        print(f"[ OK ] {label}")
        return True
    print(f"[FAIL] {label}，return code={result.returncode}")
    return False


def run_interactive(
    label: str,
    user_input: str,
    expected: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> bool:
    print(f"\n[TEST] {label}")
    result = subprocess.run(
        [sys.executable, "privacy_tool.py"],
        cwd=ROOT,
        input=user_input,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    missing = [text for text in expected if text not in output]
    unexpected = [text for text in forbidden if text in output]
    if result.returncode == 0 and not missing and not unexpected:
        print(f"[ OK ] {label}")
        return True
    print(f"[FAIL] {label}，return code={result.returncode}")
    if missing:
        print(f"       缺少输出：{missing}")
    if unexpected:
        print(f"       不应出现：{unexpected}")
    return False


def main() -> int:
    print("=== 隐私实验集成工具交付自检 ===")
    failures = 0
    for path in REQUIRED_PATHS:
        if path.exists():
            print(f"[ OK ] {path.relative_to(ROOT)}")
        else:
            print(f"[FAIL] 缺少 {path.relative_to(ROOT)}")
            failures += 1

    if failures:
        print(f"\n目录检查失败：缺少 {failures} 项。请把交付包解压到完整 x73 项目根目录。")
        return 1

    from privacy_tools.bootstrap import environment_plan
    from privacy_tools.unified import METHODS

    for method_id, method in METHODS.items():
        for experiment in method.experiments:
            plan = environment_plan(method_id, experiment.id)
            if plan.requirements is not None and not plan.requirements.exists():
                print(f"[FAIL] 缺少自动环境依赖文件：{plan.requirements}")
                failures += 1
    if failures:
        return 1
    print("[ OK ] 六种方法的全部运行项均有依赖检查方案")

    failures += 0 if run("统一入口菜单清单", [sys.executable, "privacy_tool.py", "list"]) else 1
    failures += 0 if run("30 个运行项完整结构审计", [sys.executable, "full_audit.py"]) else 1

    removed_prompts = (
        "输入模型路径",
        "输入数据集名称",
        "其他参数设置",
        "请选择 Python 环境准备方式",
    )
    failures += 0 if run_interactive(
        "新 UI 微调路径",
        "1\n1\n1\n1\n1\n1\n1\ny\n",
        ("任务：微调", "实验：当前微调方法", "Llama-2-7b", "sst2", "查看最近历史结果使用默认配置的 Python环境", "数据集路径："),
        removed_prompts,
    ) else 1
    failures += 0 if run_interactive(
        "新 UI 评估路径",
        "2\n1\n1\n2\n1\n1\n1\ny\n",
        ("任务：评估", "评估类型：直接评估", "微调类型：显式微调模型", "Llama-2-7b"),
        removed_prompts + ("数据集路径：",),
    ) else 1
    failures += 0 if run_interactive(
        "ChatGLM 显式无扰动评估接口",
        "2\n1\n1\n3\n2\n1\n",
        ("模型：ChatGLM3-6b", "ChatGLM3-6b 显式微调无扰动评估接口已预留"),
        ("请选择模型\n", "方法：Kal-DP", "方法：DADP-ICL"),
    ) else 1
    failures += 0 if run_interactive(
        "Llama 显式无扰动直接评估",
        "2\n1\n1\n3\n1\n1\ny\n",
        ("模型：Llama-2-7b", "扰动类型：无扰动", "===== 显式微调 直接评估结果 =====", "基准方法"),
        ("请选择模型\n", "方法：Kal-DP", "本方法\n"),
    ) else 1
    failures += 0 if run_interactive(
        "Llama 隐式无扰动间接评估",
        "2\n2\n2\n2\n1\n1\ny\n",
        ("模型：Llama-2-7b", "扰动类型：无扰动", "===== 隐式微调 间接评估结果 =====", "基准方法"),
        ("请选择模型\n", "方法：DADP-ICL", "本方法\n"),
    ) else 1
    failures += 0 if run_interactive(
        "Llama 隐式无扰动直接评估",
        "2\n1\n2\n2\n1\n1\ny\n",
        ("按原始说明用原始示例提示执行前缀抽取", "按原始 member_test 统计原始提示攻击命中率"),
        ("按原始说明用合成示例提示执行前缀抽取", "统计合成提示攻击命中率"),
    ) else 1

    for method in ("cliptext", "ptcds", "kal_dp", "zo_dpga", "dadp_icl", "privsv"):
        for experiment in ("utility", "baseline", "performance_baseline", "direct_attack", "indirect_attack"):
            command = [sys.executable, "privacy_tool.py", "run", "--method", method, "--experiment", experiment, "--dry-run"]
            failures += 0 if run(f"{method}/{experiment} 命令预演", command) else 1

    if failures:
        print(f"\n自检未通过，共 {failures} 项失败。")
        return 1
    print("\n自检通过：微调/评估任务菜单、六种方法的 30 个底层运行项、统一指标 CSV，以及全部命令预演均正常。")
    print("下一步：直接运行 python3 privacy_tool.py，按助手提示选择任务和方法。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
