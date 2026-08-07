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
    print("[ OK ] 六种方法的全部运行项均有自动环境方案")

    failures += 0 if run("统一入口菜单清单", [sys.executable, "privacy_tool.py", "list"]) else 1
    failures += 0 if run("24 个运行项完整结构审计", [sys.executable, "full_audit.py"]) else 1

    for method in ("cliptext", "ptcds", "kal_dp", "zo_dpga", "dadp_icl", "privsv"):
        for experiment in ("utility", "baseline", "direct_attack", "indirect_attack"):
            command = [sys.executable, "privacy_tool.py", "run", "--method", method, "--experiment", experiment, "--dry-run"]
            failures += 0 if run(f"{method}/{experiment} 命令预演", command) else 1

    if failures:
        print(f"\n自检未通过，共 {failures} 项失败。")
        return 1
    print("\n自检通过：三级类别菜单、六种方法的 24 个运行项、统一指标 CSV，以及全部训练/对比/直接评估/间接评估命令预演均正常。")
    print("下一步：直接运行 python3 privacy_tool.py，按助手提示选择方法和实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
