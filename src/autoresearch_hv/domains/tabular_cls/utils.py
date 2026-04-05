"""Naming helpers for the tabular classification domain."""

from __future__ import annotations


def version_stem(version: str) -> str:
    """Build the canonical file stem, e.g. 'v1.0_Tabular_CLS'."""
    return f"{version}_Tabular_CLS"


def version_slug(version: str) -> str:
    """Build a filesystem-safe slug, e.g. 'v1-0-tabular-cls'."""
    return version.replace(".", "-").lower() + "-tabular-cls"
