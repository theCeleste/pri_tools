"""Check package integrity and pinned dependency versions without using a GPU."""
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
    parser.add_argument("--check-model", action="store_true", help="Hash external ChatGLM model files")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checksums.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Package file changed: {name}")
    print(f"SHA256 passed: {len(manifest['files'])} files", flush=True)
    for line in ["torch==2.5.1", *(ROOT / "requirements.txt").read_text().splitlines()]:
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Dependency mismatch: {name} expected={expected} actual={actual}")
    print("Pinned dependency versions passed", flush=True)
    if args.check_model:
        model = Path(os.environ.get("CHATGLM_MODEL_DIR", ROOT.parent / "models/chatglm3-6b"))
        if not model.is_absolute():
            model = ROOT / model
        for name, expected in manifest["model_files"].items():
            if digest(model / name) != expected:
                raise ValueError(f"External model changed: {name}")
        print("External ChatGLM SHA256 passed", flush=True)


if __name__ == "__main__":
    main()
