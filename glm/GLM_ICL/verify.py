"""Verify delivery hashes and optionally the external ChatGLM model snapshot."""
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
    parser.add_argument("--check-model", action="store_true", help="Hash all external model files (about 12 GB)")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "checksums.json").read_text())
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
        model = Path(os.environ.get("CHATGLM_MODEL_DIR", ROOT.parent / "models/chatglm3-6b"))
        for name, expected in manifest["model_files"].items():
            if digest(model / name) != expected:
                raise ValueError(f"External model changed: {name}")
        print("External ChatGLM SHA256 passed")


if __name__ == "__main__":
    main()
