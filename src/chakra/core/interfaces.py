"""Protocol definitions that every research domain must implement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VersionPaths:
    """Canonical file locations for a versioned research run."""

    root: Path
    version: str
    stem: str
    notebook: Path
    doc: Path
    review: Path
    review_json: Path
    configs: dict[str, Path]
    notebook_readme: Path
    kernel_metadata: Path
    next_ablation: Path
    run_manifest: Path
    kaggle_output_dir: Path


@runtime_checkable
class DomainLifecycleHooks(Protocol):
    """Contract every domain must fulfil to plug into the generic lifecycle."""

    def version_stem(self, version: str) -> str:
        """Convert a version label into the canonical notebook stem."""
        ...

    def version_slug(self, version: str) -> str:
        """Build a filesystem and Kaggle-safe slug from a version label."""
        ...

    def resolve_version_paths(self, version: str, root: Path | None = None) -> VersionPaths:
        """Compute canonical file paths for a version."""
        ...

    def build_version_configs(self, version: str, parent: str | None, lineage: str) -> dict[str, dict[str, Any]]:
        """Build the set of config variants (control, smoke, train, etc.) for a version."""
        ...

    def render_notebook(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        """Render a Kaggle-ready notebook JSON string."""
        ...

    def render_doc(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        """Render the external run documentation markdown."""
        ...

    def render_review(self, version: str) -> str:
        """Render the initial review/roast template."""
        ...

    def render_notebook_readme(self) -> str:
        """Render the README for the versioned notebooks directory."""
        ...

    def default_kernel_metadata(self) -> dict[str, Any]:
        """Return default Kaggle kernel metadata for the domain."""
        ...

    def build_findings(
        self, version: str, manifest: dict[str, Any], eval_summary: dict[str, Any] | None, registry: dict[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, float]]:
        """Analyse a synced run and return severity-ordered findings + metric deltas."""
        ...

    def ablation_suggestions(
        self, version: str, eval_summary: dict[str, Any] | None, delta: dict[str, float],
    ) -> list[str]:
        """Suggest bounded next-version ablations."""
        ...

    def roast_lines(self) -> list[str]:
        """Return domain-specific roast / audit lines."""
        ...

    def validate_version(self, version: str) -> list[str]:
        """Run domain-specific validation and return a list of failure messages (empty = pass)."""
        ...
