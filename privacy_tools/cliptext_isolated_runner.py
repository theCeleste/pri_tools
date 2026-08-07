from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="在单次结果目录中隔离运行 ClipText")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--eps", required=True)
    parser.add_argument("--top-k", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    source = args.source_root.resolve()
    workspace = args.workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f"隔离工作目录已存在：{workspace}")
    workspace.mkdir(parents=True)

    for path in source.glob("*.py"):
        shutil.copy2(path, workspace / path.name)
    shutil.copytree(source / "datasets", workspace / "datasets")
    for dirname in ("p_dict", "sim_word_dict"):
        relative = Path("glove_840B-300d") / "conservative" / f"eps_{args.eps}_top_{args.top_k}.txt"
        candidate = source / dirname / relative
        if candidate.is_file():
            destination = workspace / dirname / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)

    mapping = workspace / "p_dict" / "glove_840B-300d" / "conservative" / f"eps_{args.eps}_top_{args.top_k}.txt"
    similarity = workspace / "sim_word_dict" / "glove_840B-300d" / "conservative" / f"eps_{args.eps}_top_{args.top_k}.txt"
    if not (mapping.is_file() and similarity.is_file()):
        glove = source / "embeddings" / "glove_840B-300d.txt"
        if not glove.is_file():
            raise FileNotFoundError("当前 epsilon/top-k 没有预计算映射，且未找到 GloVe；只有重建映射时才需要 GloVe")
        _link_or_copy(glove, workspace / "embeddings" / glove.name)

    entry_target = workspace / "integration_entry.py"
    entry_text = args.entry.read_text(encoding="utf-8")
    entry_text = entry_text.replace(
        "from transformers import AdamW,get_linear_schedule_with_warmup,BertModel,AutoConfig",
        "from torch.optim import AdamW\nfrom transformers import get_linear_schedule_with_warmup,BertModel,AutoConfig",
    )
    entry_target.write_text(entry_text, encoding="utf-8")
    passthrough = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, str(entry_target), *passthrough]
    print(f"ClipText 隔离工作目录：{workspace}")
    print("原始源码目录仅作为只读输入，不会保存本次数据、日志或模型。")
    return_code = subprocess.run(command, cwd=workspace).returncode
    outputs = []
    for path in sorted((workspace / "privatized_dataset").rglob("*.tsv")) if (workspace / "privatized_dataset").exists() else []:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "source_root": str(source),
        "isolated_workspace": str(workspace),
        "source_tree_written": False,
        "return_code": return_code,
        "generated_datasets": outputs,
    }
    (workspace.parent / "cliptext_outputs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
