from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .core import PROJECT_ROOT


ENV_ROOT = PROJECT_ROOT / ".privacy_envs"
DOWNLOAD_ROOT = PROJECT_ROOT / ".privacy_downloads"
REQUIREMENTS_ROOT = Path(__file__).resolve().parent / "requirements" / "auto"
GLOVE_TARGET = PROJECT_ROOT / "ClipText" / "sst" / "ClipText" / "ClipText" / "sst2" / "embeddings" / "glove_840B-300d.txt"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"


def clip_mapping_available(epsilon: object, top_k: object) -> bool:
    """Return whether ClipText can run with the shipped mapping and no GloVe vectors."""
    root = GLOVE_TARGET.parent.parent
    filename = f"eps_{epsilon}_top_{top_k}.txt"
    relative = Path("glove_840B-300d") / "conservative" / filename
    return (root / "p_dict" / relative).is_file() and (root / "sim_word_dict" / relative).is_file()


@dataclass(frozen=True)
class EnvironmentPlan:
    level: str
    requirements: Path | None
    imports: tuple[str, ...]
    explanation: str


FULL_IMPORTS: dict[str, tuple[str, ...]] = {
    "cliptext": ("torch", "transformers", "datasets", "peft", "numpy", "pandas", "sklearn", "nltk"),
    "ptcds": ("torch", "transformers", "datasets", "peft", "numpy", "pandas", "sklearn"),
    "kal_dp": ("torch", "transformers", "datasets", "peft", "opacus", "numpy", "pandas", "sklearn"),
    "zo_dpga": ("torch", "transformers", "datasets", "peft", "opacus", "sklearn", "yaml"),
    "dadp_icl": ("torch", "transformers", "datasets", "autodp", "scipy", "pandas", "tiktoken"),
    "privsv": ("torch", "transformers", "datasets", "steering_vectors", "sklearn", "pandas"),
}
FULL_IMPORTS["dpsgd"] = ("torch", "transformers", "datasets", "peft", "opacus", "numpy", "pandas", "sklearn", "psutil")
FULL_IMPORTS["glm_subj"] = ("torch", "transformers", "peft", "opacus", "accelerate", "safetensors", "numpy", "sklearn", "tqdm")

FULL_REQUIREMENTS = {
    method: REQUIREMENTS_ROOT / f"{method}.txt" for method in FULL_IMPORTS
}
FULL_REQUIREMENTS["dpsgd"] = REQUIREMENTS_ROOT / "dpsgd.txt"
FULL_REQUIREMENTS["glm_subj"] = PROJECT_ROOT / "Glm-subj" / "requirements.txt"

NO_DEPENDENCY_EXPERIMENTS = {
    ("zo_dpga", "summarize"),
    ("privsv", "direct_existing"),
}

LIGHT_EXPERIMENTS = {
    ("cliptext", "direct_existing"),
    ("cliptext", "indirect_existing"),
    ("ptcds", "direct_existing"),
    ("ptcds", "indirect_existing"),
    ("kal_dp", "direct_existing"),
    ("kal_dp", "indirect_existing"),
    ("dadp_icl", "direct_existing"),
    ("dadp_icl", "indirect_existing"),
    ("privsv", "indirect_existing"),
}


def environment_plan(method_id: str, experiment_id: str) -> EnvironmentPlan:
    pair = (method_id, experiment_id)
    if pair in NO_DEPENDENCY_EXPERIMENTS:
        return EnvironmentPlan("none", None, (), "该实验只使用 Python 标准库或集成层已有结果。")
    if pair in LIGHT_EXPERIMENTS:
        return EnvironmentPlan(
            "light",
            REQUIREMENTS_ROOT / "analysis.txt",
            ("numpy", "pandas", "sklearn", "matplotlib"),
            "该实验只需要常用的数据分析依赖，不需要安装训练栈。",
        )
    return EnvironmentPlan(
        "full",
        FULL_REQUIREMENTS[method_id],
        FULL_IMPORTS[method_id],
        "该实验需要该方法的完整 PyTorch/Transformers 运行环境。",
    )


def environment_python(method_id: str, level: str) -> Path:
    name = "analysis" if level == "light" else method_id
    root = ENV_ROOT / name
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment_root(python_path: Path) -> Path:
    return python_path.parent.parent


def check_imports(python_path: str | Path, imports: tuple[str, ...]) -> tuple[bool, list[str]]:
    if not imports:
        return True, []
    executable = str(python_path)
    if not Path(executable).exists() and shutil.which(executable) is None:
        return False, ["Python 解释器不存在"]
    code = (
        "import importlib.util, json; "
        f"names={list(imports)!r}; "
        "print(json.dumps([n for n in names if importlib.util.find_spec(n) is None]))"
    )
    result = subprocess.run([executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return False, [result.stderr.strip() or "无法检查依赖"]
    try:
        import json

        missing = json.loads(result.stdout.strip())
    except Exception:
        return False, ["依赖检查输出无法解析"]
    return not missing, [str(item) for item in missing]


def check_cuda(python_path: str | Path, gpu: str) -> tuple[bool, str]:
    code = (
        "import json, torch; "
        "print(json.dumps({'available': torch.cuda.is_available(), "
        "'count': torch.cuda.device_count(), "
        "'torch': torch.__version__, "
        "'cuda': torch.version.cuda}))"
    )
    result = subprocess.run([str(python_path), "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return False, result.stderr.strip() or "无法导入 PyTorch 检查 GPU"
    try:
        import json

        info = json.loads(result.stdout.strip())
        indices = [int(part.strip()) for part in str(gpu).split(",") if part.strip()]
    except (ValueError, TypeError, KeyError):
        return False, f"GPU 编号无效：{gpu}"
    if not indices:
        return False, f"GPU 编号无效：{gpu}"
    if not info.get("available"):
        return False, f"PyTorch {info.get('torch')} 未检测到 CUDA（torch CUDA={info.get('cuda')}）"
    count = int(info.get("count", 0))
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        return False, f"GPU 编号 {invalid} 超出范围；当前检测到 {count} 张 GPU"
    return True, f"PyTorch {info.get('torch')}，CUDA {info.get('cuda')}，使用 GPU {indices}（共检测到 {count} 张）"


def ensure_nltk_stopwords(python_path: str | Path) -> int:
    check = subprocess.run(
        [str(python_path), "-c", "import nltk; nltk.data.find('corpora/stopwords')"],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return 0
    print("正在下载 ClipText 使用的 NLTK stopwords 小型语料……")
    return subprocess.run([str(python_path), "-m", "nltk.downloader", "stopwords"]).returncode


def create_environment(method_id: str, plan: EnvironmentPlan) -> tuple[int, Path]:
    """Create/reuse a managed environment and install the selected runtime."""
    python_path = environment_python(method_id, plan.level)
    root = _environment_root(python_path)
    root.parent.mkdir(parents=True, exist_ok=True)
    if not python_path.exists():
        conda = shutil.which("conda")
        if conda:
            command = [conda, "create", "--prefix", str(root), "python=3.10", "pip", "-y"]
        else:
            if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
                print(
                    f"当前 Python 为 {sys.version_info.major}.{sys.version_info.minor}，且未找到 conda；"
                    "原始实验依赖要求 Python 3.10 或 3.11。请安装 conda，或用 Python 3.10/3.11 启动工具。"
                )
                return 2, python_path
            command = [sys.executable, "-m", "venv", str(root)]
        print("\n正在创建隔离环境：")
        print("  " + " ".join(command))
        code = subprocess.run(command).returncode
        if code != 0:
            return code, python_path
    if plan.requirements is None:
        return 0, python_path
    print("\n正在安装/补齐依赖。这一步可能需要较长时间，只在首次准备时执行。")
    upgrade = subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]).returncode
    if upgrade != 0:
        return upgrade, python_path
    install = subprocess.run([str(python_path), "-m", "pip", "install", "-r", str(plan.requirements)]).returncode
    return install, python_path


def copy_glove(source: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 GloVe 文件：{source}")
    GLOVE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, GLOVE_TARGET)


def download_glove() -> None:
    """Download the official 840B archive and extract only the required vector file."""
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    GLOVE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOAD_ROOT / "glove.840B.300d.zip"
    free = shutil.disk_usage(PROJECT_ROOT).free
    required = 9 * 1024**3
    if free < required:
        raise OSError(f"可用空间不足：至少需要约 9 GiB，当前只有 {free / 1024**3:.1f} GiB")

    last_percent = -1

    def progress(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_percent
        if total <= 0:
            return
        percent = min(100, int(blocks * block_size * 100 / total))
        if percent >= last_percent + 5:
            print(f"  下载进度：{percent}%")
            last_percent = percent

    print(f"正在从 Stanford NLP 下载约 2.03 GiB：{GLOVE_URL}")
    urllib.request.urlretrieve(GLOVE_URL, archive, progress)
    print("正在解压所需的 glove.840B.300d.txt（解压后约 5.3 GiB）……")
    with zipfile.ZipFile(archive) as bundle:
        member = next((name for name in bundle.namelist() if name.endswith("glove.840B.300d.txt")), None)
        if member is None:
            raise OSError("下载包中没有找到 glove.840B.300d.txt")
        with bundle.open(member) as source, GLOVE_TARGET.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
