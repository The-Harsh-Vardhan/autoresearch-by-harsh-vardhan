"""HNDSR-specific utilities."""

from __future__ import annotations

import re

# Domain-specific helpers — shared infrastructure comes from core.utils
VERSION_PATTERN = re.compile(r"^vR(?:\.P)?\.\d+(?:\.\d+)?$")


def version_stem(version: str) -> str:
    """Convert a version label into the canonical notebook stem."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Unsupported HNDSR version label: {version}")
    return f"{version}_HNDSR"


def version_slug(version: str) -> str:
    """Build a filesystem and Kaggle-safe slug from a version label."""
    normalized = version.lower().replace(".", "").replace("_", "-")
    return f"{normalized}-hndsr"
