"""Read and write the latest no-perturbation evaluation result."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

from .core import project_path

SCHEMA_VERSION = 1


def history_dir(settings: dict, evaluation_type: str, ident: dict) -> Path:
    """Return the single latest-result directory for one evaluation identity."""
    root = project_path(str(settings.get("historical_results_dir", "privacy_tools/historical_results")))
    return root / "no_perturbation" / evaluation_type / ident["model"] / ident["finetune_type"] / ident["dataset"]


def archive_result(settings: dict, evaluation_type: str, summary: dict, result_text: str) -> Path:
    """Atomically replace the latest archived JSON and text result."""
    ident = {key: summary[key] for key in ("model", "finetune_type", "dataset")}
    target = history_dir(settings, evaluation_type, ident)
    target.mkdir(parents=True, exist_ok=True)

    payload = dict(summary)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["baseline_type"] = "no_perturbation"
    payload["evaluation_type"] = evaluation_type
    payload.setdefault("updated_at", payload.get("completed_at") or datetime.now(timezone.utc).isoformat())

    token = uuid4().hex
    json_tmp = target / f".summary.{token}.tmp"
    text_tmp = target / f".result.{token}.tmp"
    try:
        json_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        text_tmp.write_text(result_text, encoding="utf-8")
        # Replace the text first and the identity-bearing JSON last. Readers
        # therefore continue to see the previous complete summary during an
        # interrupted update and can fall back to the new text if necessary.
        os.replace(text_tmp, target / "result.txt")
        os.replace(json_tmp, target / "summary.json")
    finally:
        for temporary in (json_tmp, text_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return target


def load_result(settings: dict, evaluation_type: str, ident: dict) -> dict | None:
    """Load structured history, or return the raw text fallback."""
    target = history_dir(settings, evaluation_type, ident)
    summary_file = target / "summary.json"
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            required = ("model", "finetune_type", "dataset", "engine")
            matches_identity = all(summary.get(key) == ident.get(key) for key in required)
            structured = (
                summary.get("schema_version") == SCHEMA_VERSION
                and summary.get("baseline_type") == "no_perturbation"
                and summary.get("evaluation_type") == evaluation_type
                and summary.get("status") == "completed"
                and matches_identity
                and isinstance(summary.get("results"), dict)
            )
            if structured:
                return {"kind": "summary", "path": target, "summary": summary}
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        pass

    text_file = target / "result.txt"
    try:
        return {"kind": "text", "path": target, "text": text_file.read_text(encoding="utf-8")}
    except (OSError, UnicodeError):
        return None
