"""Utilities for the HNDSR scratch lane."""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    import yaml
except Exception:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_PATTERN = re.compile(r"^vR(?:\.P)?\.\d+(?:\.\d+)?$")


def repo_path(value: str | Path) -> Path:
    """Resolve a repo-relative path without guessing from cwd."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    target = repo_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_workspace_temp(root: str | Path) -> Path:
    """Force temporary files into the workspace to avoid broken system temp permissions."""
    temp_root = ensure_dir(Path(root) / ".tmp" / "temp")
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMPDIR"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    return temp_root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional `inherits` support."""
    resolved = repo_path(path)
    text = resolved.read_text(encoding="utf-8")
    if yaml is not None:
        config = yaml.safe_load(text) or {}
    else:
        config = json.loads(text)
    parent = config.pop("inherits", None)
    if not parent:
        return config
    base = load_config(parent)
    return _deep_merge(base, config)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON payload with stable formatting."""
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON payload from disk."""
    resolved = repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, content: str) -> Path:
    """Persist UTF-8 text content."""
    resolved = repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return resolved


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(explicit: str | None = None) -> torch.device:
    """Pick a device conservatively for reproducible experiments."""
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config for tracker logging."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        scoped = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, scoped))
        else:
            flat[scoped] = value
    return flat


def describe_run_dirs(config: dict[str, Any], run_name: str) -> dict[str, Path]:
    """Compute the tracked output directories for a run."""
    artifact_root = ensure_dir(config["paths"]["artifact_root"])
    run_root = artifact_root / run_name
    return {
        "artifact_root": artifact_root,
        "run_root": ensure_dir(run_root),
        "checkpoints": ensure_dir(run_root / "checkpoints"),
        "metrics": ensure_dir(run_root / "metrics"),
        "samples": ensure_dir(run_root / "samples"),
        "tracker": ensure_dir(run_root / "tracker"),
        "manifests": ensure_dir(run_root / "manifests"),
    }


def version_stem(version: str) -> str:
    """Convert a version label into the canonical notebook stem."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Unsupported version label: {version}")
    return f"{version}_HNDSR"


def version_slug(version: str) -> str:
    """Build a filesystem and Kaggle-safe slug from a version label."""
    normalized = version.lower().replace(".", "").replace("_", "-")
    return f"{normalized}-hndsr"

