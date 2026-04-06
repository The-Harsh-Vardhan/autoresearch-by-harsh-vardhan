"""Domain registry — discovers and loads research domain manifests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ManifestError
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
    domain: str | None = None
    task: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list[str]] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    ablation_templates: dict[str, list[str]] = field(default_factory=dict)
    lifecycle: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    agents: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainManifest:
        _validate_manifest(data)
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
            domain=data.get("domain"),
            task=data.get("task"),
            meta=data.get("meta", {}),
            capabilities=data.get("capabilities", {}),
            defaults=data.get("defaults", {}),
            metrics=data.get("metrics", {}),
            search_space=data.get("search_space", {}),
            failure_modes=data.get("failure_modes", []),
            ablation_templates=data.get("ablation_templates", {}),
            lifecycle=data.get("lifecycle", {}),
            execution=data.get("execution", {}),
            intent=data.get("intent"),
            agents=data.get("agents", {}),
        )


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"domain.yaml validation error at '{key}': expected non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"domain.yaml validation error at '{key}': expected string when provided")
    return value


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"domain.yaml validation error at '{key}': expected mapping/dict")
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"domain.yaml validation error at '{key}': expected mapping/dict")
    return value


def _optional_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"domain.yaml validation error at '{key}': expected list")
    return value


def _require_str_list(data: dict[str, Any], key: str) -> list[str]:
    values = _optional_list(data, key)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"domain.yaml validation error at '{key}': expected list[str]")
    return values


def _validate_dict_str_values(payload: dict[str, Any], path: str) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"domain.yaml validation error at '{path}': expected string keys")
        if not isinstance(value, str):
            raise ValueError(f"domain.yaml validation error at '{path}.{key}': expected string")


def _validate_dict_bool_values(payload: dict[str, Any], path: str) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"domain.yaml validation error at '{path}': expected string keys")
        if not isinstance(value, bool):
            raise ValueError(f"domain.yaml validation error at '{path}.{key}': expected boolean")


def _validate_dict_list_str_values(payload: dict[str, Any], path: str) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"domain.yaml validation error at '{path}': expected string keys")
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"domain.yaml validation error at '{path}.{key}': expected list[str]")


def _validate_manifest(data: dict[str, Any]) -> None:
    _require_str(data, "name")
    _require_str(data, "display_name")
    _require_str(data, "version_pattern")
    _require_str(data, "primary_metric")
    _require_str(data, "metric_direction")
    _require_str(data, "benchmark_registry")
    _require_str(data, "config_dir")
    _require_str(data, "programs_doc")
    _require_str_list(data, "model_kinds")
    _require_dict(data, "entrypoints")

    metric_direction = data.get("metric_direction")
    if metric_direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError(
            "domain.yaml validation error at 'metric_direction': expected one of "
            "{'higher_is_better', 'lower_is_better'}"
        )

    domain = _optional_str(data, "domain")
    if domain and domain not in {"cv", "nlp", "tb", "ts", "audio", "rl", "multi", "graph"}:
        raise ValueError(
            "domain.yaml validation error at 'domain': expected one of "
            "{'cv','nlp','tb','ts','audio','rl','multi','graph'}"
        )

    intent = _optional_str(data, "intent")
    if intent and intent not in {"optimize", "explore", "reproduce", "benchmark"}:
        raise ValueError(
            "domain.yaml validation error at 'intent': expected one of "
            "{'optimize','explore','reproduce','benchmark'}"
        )

    _optional_str(data, "task")

    meta = _optional_dict(data, "meta")
    _validate_dict_str_values(meta, "meta")

    capabilities = _optional_dict(data, "capabilities")
    _validate_dict_bool_values(capabilities, "capabilities")

    defaults = _optional_dict(data, "defaults")
    _validate_dict_str_values(defaults, "defaults")

    metrics = _optional_dict(data, "metrics")
    if metrics:
        primary = metrics.get("primary")
        secondary = metrics.get("secondary", [])
        if primary is not None and not isinstance(primary, str):
            raise ValueError("domain.yaml validation error at 'metrics.primary': expected string")
        if not isinstance(secondary, list) or not all(isinstance(item, str) for item in secondary):
            raise ValueError("domain.yaml validation error at 'metrics.secondary': expected list[str]")

    search_space = _optional_dict(data, "search_space")
    _validate_dict_list_str_values(search_space, "search_space")

    _require_str_list(data, "failure_modes") if "failure_modes" in data else _optional_list(data, "failure_modes")

    ablation_templates = _optional_dict(data, "ablation_templates")
    _validate_dict_list_str_values(ablation_templates, "ablation_templates")

    lifecycle = _optional_dict(data, "lifecycle")
    if "requires_gpu" in lifecycle and not isinstance(lifecycle["requires_gpu"], bool):
        raise ValueError("domain.yaml validation error at 'lifecycle.requires_gpu': expected boolean")
    if "supports_local" in lifecycle and not isinstance(lifecycle["supports_local"], bool):
        raise ValueError("domain.yaml validation error at 'lifecycle.supports_local': expected boolean")
    if "max_runtime" in lifecycle and not isinstance(lifecycle["max_runtime"], (int, float, str)):
        raise ValueError("domain.yaml validation error at 'lifecycle.max_runtime': expected int|float|string")

    execution = _optional_dict(data, "execution")
    if execution:
        default = execution.get("default", "auto")
        supports = execution.get("supports", ["local", "kaggle"])
        if default not in {"local", "kaggle", "auto"}:
            raise ValueError("domain.yaml validation error at 'execution.default': expected one of {'local','kaggle','auto'}")
        if not isinstance(supports, list) or not supports:
            raise ValueError("domain.yaml validation error at 'execution.supports': expected non-empty list")
        if not all(item in {"local", "kaggle", "auto"} for item in supports):
            raise ValueError(
                "domain.yaml validation error at 'execution.supports': expected values from {'local','kaggle','auto'}"
            )

    agents = _optional_dict(data, "agents")
    _validate_dict_str_values(agents, "agents")


def load_domain_strategy(manifest: DomainManifest) -> Any:
    """Resolve and instantiate a deterministic strategy implementation for a domain."""
    from .domain_strategy import CVStrategy, GenericStrategy, NLPStrategy

    if manifest.domain == "nlp" or manifest.name.startswith("nlp"):
        return NLPStrategy(manifest)
    if manifest.domain == "cv" or manifest.name.startswith("hndsr"):
        return CVStrategy(manifest)
    return GenericStrategy(manifest)


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
