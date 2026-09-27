"""Verify delivery hashes and optionally one external target-model snapshot."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fm9g", "chatglm3"), default="fm9g")
    parser.add_argument("--check-model", action="store_true", help="Hash every file in the selected external model")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checksums.json").read_text())
    actual_delivery = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                       if p.is_file() and p.name != "checksums.json"
                       and "runs" not in p.relative_to(ROOT).parts
                       and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual_delivery != set(manifest["files"]):
        raise ValueError("Delivery file list differs from checksums.json")
    for name, expected in manifest["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Delivery file changed: {name}")
    print(f"Delivery SHA256 passed: {len(manifest['files'])} files")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        package, expected = line.split("==")
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Dependency mismatch: {package} expected={expected} actual={actual}")
    print("Pinned dependency versions passed")
    if args.check_model:
        environment = "FM9G_MODEL_DIR" if args.model == "fm9g" else "CHATGLM_MODEL_DIR"
        default = "models/9G7B_MHA" if args.model == "fm9g" else "models/chatglm3-6b"
        model = Path(os.environ.get(environment, ROOT.parent / default))
        if not model.is_dir():
            raise FileNotFoundError(model)
        expected_files = manifest["model_files"][args.model]
        actual_files = {p.relative_to(model).as_posix() for p in model.rglob("*")
                        if p.is_file() and ".cache" not in p.relative_to(model).parts
                        and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if actual_files != set(expected_files):
            raise ValueError(f"External {args.model} file list differs from the verified snapshot")
        for name, expected in expected_files.items():
            if digest(model / name) != expected:
                raise ValueError(f"External model changed: {name}")
        print(f"External {args.model} SHA256 passed: {len(expected_files)} files")


if __name__ == "__main__":
    main()
