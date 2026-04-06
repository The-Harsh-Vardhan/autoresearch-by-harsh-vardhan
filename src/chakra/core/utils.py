"""Domain-agnostic utilities for AutoResearch."""

from __future__ import annotations

import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Workspace resolution — works for both git-clone and pip-install users
# ---------------------------------------------------------------------------

def _find_workspace_root() -> Path:
    """Locate the Chakra workspace root directory.

    Resolution order:
      1. ``CHAKRA_WORKSPACE`` environment variable (explicit override)
      2. Walk up from *cwd()* looking for ``pyproject.toml``
      3. Fall back to *cwd()* (assume user is in their project directory)
    """
    env_root = os.environ.get("CHAKRA_WORKSPACE")
    if env_root:
        candidate = Path(env_root).resolve()
        if candidate.is_dir():
            return candidate

    # Walk up from cwd looking for a pyproject.toml
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
        # Stop at filesystem root
        if parent == parent.parent:
            break

    return Path.cwd().resolve()


REPO_ROOT = _find_workspace_root()


def load_dotenv(root: Path | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (no external dep)."""
    env_path = (root or REPO_ROOT) / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


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


def load_yaml_text(text: str) -> dict[str, Any]:
    """Parse YAML text, falling back to a minimal key-value parser if pyyaml is unavailable."""
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # Minimal fallback: try JSON first, then a simple line-based YAML parser
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML parser for flat / single-nested maps with list support.

    Handles the structures used in domain.yaml and config files when pyyaml
    is not installed: top-level scalars, nested dicts (one level), and lists
    declared with ``- item`` syntax or ``[inline, list]`` syntax.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_value: dict[str, Any] | list[Any] | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())

        # Dash-list item: "  - value"
        if stripped.startswith("- "):
            item_val = stripped[2:].strip()
            if current_key is not None:
                # Convert current_value to list if it isn't already
                if not isinstance(current_value, list):
                    current_value = []
                    result[current_key] = current_value
                current_value.append(_yaml_scalar(item_val))
            continue

        if ":" not in stripped:
            continue

        key_part, _, val_part = stripped.partition(":")
        key = key_part.strip()
        val = val_part.strip()

        if indent == 0:
            if val:
                if val.startswith("[") and val.endswith("]"):
                    # Inline list at top level
                    result[key] = [_yaml_scalar(item.strip()) for item in val[1:-1].split(",") if item.strip()]
                else:
                    result[key] = _yaml_scalar(val)
                current_key = None
                current_value = None
            else:
                # Start of a nested block (could be dict or list — determined by children)
                current_key = key
                current_value = {}
                result[key] = current_value
        elif indent > 0 and isinstance(current_value, dict):
            if val.startswith("[") and val.endswith("]"):
                items = [_yaml_scalar(item.strip()) for item in val[1:-1].split(",") if item.strip()]
                current_value[key] = items
            elif val:
                current_value[key] = _yaml_scalar(val)
            else:
                current_value[key] = {}

    return result


def _yaml_scalar(val: str) -> Any:
    """Convert a YAML scalar string to a Python value."""
    # Remove surrounding quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    lower = val.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    if lower in ("null", "none", "~"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional ``inherits`` support."""
    resolved = repo_path(path)
    text = resolved.read_text(encoding="utf-8")
    config = load_yaml_text(text)
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
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(explicit: str | None = None) -> Any:
    """Pick a device conservatively for reproducible experiments.

    Returns a ``torch.device`` when torch is available, or a plain string
    (``"cpu"``) when it is not.
    """
    try:
        import torch
    except ImportError:
        return explicit or "cpu"
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
