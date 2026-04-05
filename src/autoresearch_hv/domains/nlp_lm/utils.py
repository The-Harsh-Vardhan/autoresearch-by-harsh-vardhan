"""NLP domain-specific utilities."""

from __future__ import annotations

import re

VERSION_PATTERN = re.compile(r"^v\d+\.\d+(?:\.\d+)?$")


def version_stem(version: str) -> str:
    """Convert a version label into the canonical notebook stem."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Unsupported NLP version label: {version}")
    return f"{version}_NLP_LM"


def version_slug(version: str) -> str:
    """Build a filesystem and Kaggle-safe slug from a version label."""
    normalized = version.lower().replace(".", "").replace("_", "-")
    return f"{normalized}-nlp-lm"
