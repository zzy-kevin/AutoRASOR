"""Machine-readable output helpers for AutoRASOR runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_text(Path(path), json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    text = "".join(json.dumps(json_safe(row), sort_keys=True) + "\n" for row in rows)
    _atomic_text(Path(path), text)


def save_run(output_dir: Path, result: Mapping[str, Any]) -> Path:
    """Save a controller result without generating plots."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", result.get("config", {}))
    write_jsonl(root / "observations.jsonl", result.get("observations", []))
    write_jsonl(root / "selections.jsonl", result.get("selections", []))
    write_jsonl(root / "metrics.jsonl", result.get("metrics", []))
    return root
