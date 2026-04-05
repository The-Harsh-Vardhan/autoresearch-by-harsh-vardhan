"""Domain registry — discovers and loads research domain manifests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interfaces import DomainLifecycleHooks
from .utils import load_yaml_text

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"


@dataclass(frozen=True)
class DomainManifest:
    """Parsed domain manifest from a ``domain.yaml`` file."""

    name: str
    display_name: str
    version_pattern: str
    model_kinds: list[str]
    primary_metric: str
    metric_direction: str  # "higher_is_better" or "lower_is_better"
    benchmark_registry: str
    config_dir: str
    programs_doc: str
    entrypoints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainManifest:
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            version_pattern=data["version_pattern"],
            model_kinds=data.get("model_kinds", []),
            primary_metric=data["primary_metric"],
            metric_direction=data.get("metric_direction", "higher_is_better"),
            benchmark_registry=data.get("benchmark_registry", ""),
            config_dir=data.get("config_dir", ""),
            programs_doc=data.get("programs_doc", ""),
            entrypoints=data.get("entrypoints", {}),
        )


def _parse_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return load_yaml_text(text)


def discover_domains(root: Path | None = None) -> dict[str, DomainManifest]:
    """Scan ``domains/`` for subdirectories containing ``domain.yaml``."""
    search_root = root or DOMAINS_ROOT
    manifests: dict[str, DomainManifest] = {}
    if not search_root.is_dir():
        return manifests
    for child in sorted(search_root.iterdir()):
        manifest_path = child / "domain.yaml"
        if child.is_dir() and manifest_path.is_file():
            data = _parse_yaml(manifest_path)
            manifest = DomainManifest.from_dict(data)
            manifests[manifest.name] = manifest
    return manifests


def get_domain(name: str, root: Path | None = None) -> DomainManifest:
    """Look up a single domain by name, raising ``KeyError`` if missing."""
    domains = discover_domains(root)
    if name not in domains:
        available = ", ".join(sorted(domains)) or "(none)"
        raise KeyError(f"Unknown domain '{name}'. Available: {available}")
    return domains[name]


def load_lifecycle_hooks(manifest: DomainManifest) -> DomainLifecycleHooks:
    """Import and instantiate the lifecycle hooks class for a domain."""
    module_path = manifest.entrypoints.get("lifecycle")
    if not module_path:
        raise ValueError(f"Domain '{manifest.name}' has no lifecycle entrypoint.")
    module = importlib.import_module(module_path)
    hooks_cls = getattr(module, "LifecycleHooks", None)
    if hooks_cls is None:
        raise AttributeError(f"Module '{module_path}' must export a 'LifecycleHooks' class.")
    return hooks_cls()
