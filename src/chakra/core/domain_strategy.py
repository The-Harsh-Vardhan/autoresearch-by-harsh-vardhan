"""Deterministic domain strategy helpers for model, metric, and ablation planning."""

from __future__ import annotations

from typing import Any

from .domain_registry import DomainManifest


class DomainStrategy:
    """Base deterministic strategy API shared by all domains."""

    def __init__(self, manifest: DomainManifest) -> None:
        self.manifest = manifest

    def suggest_model(self, config: dict[str, Any]) -> str:
        """Choose a model family using manifest defaults and fallback heuristics."""
        configured = config.get("model", {}).get("kind")
        if isinstance(configured, str) and configured:
            return configured

        default_family = self.manifest.defaults.get("model_family")
        if default_family:
            return default_family

        if self.manifest.model_kinds:
            return self.manifest.model_kinds[0]

        return "baseline"

    def suggest_metrics(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return metric priorities for execution and review."""
        metrics = dict(self.manifest.metrics)
        if "primary" not in metrics:
            metrics["primary"] = self.manifest.primary_metric
        if "secondary" not in metrics:
            metrics["secondary"] = []
        return metrics

    def generate_ablation_plan(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Create a deterministic ablation plan from manifest templates."""
        version = config.get("runtime", {}).get("version", "unknown")
        templates = self.manifest.ablation_templates
        plan: list[dict[str, Any]] = []

        for axis in ("architecture", "optimization", "data"):
            for idx, change in enumerate(templates.get(axis, []), start=1):
                plan.append(
                    {
                        "id": f"{version}:{axis}:{idx}",
                        "axis": axis,
                        "change": change,
                        "intent": self.manifest.intent or "optimize",
                    }
                )

        if plan:
            return plan

        fallback_model = self.suggest_model(config)
        return [
            {
                "id": f"{version}:architecture:1",
                "axis": "architecture",
                "change": f"model.kind={fallback_model}",
                "intent": self.manifest.intent or "explore",
            }
        ]

    def detect_failure_modes(self, logs: dict[str, Any]) -> list[str]:
        """Detect failure signals using manifest-declared failure mode keywords."""
        failure_modes = self.manifest.failure_modes
        if not failure_modes:
            return []

        corpus = "\n".join(
            [
                str(logs.get("error", "")),
                str(logs.get("stderr", "")),
                str(logs.get("status", "")),
            ]
        ).lower()

        detected: list[str] = []
        for mode in failure_modes:
            mode_key = mode.lower()
            if mode_key in corpus:
                detected.append(mode)
        return detected


class NLPStrategy(DomainStrategy):
    """Rule-based strategy tuned for NLP language or generation workloads."""

    def suggest_model(self, config: dict[str, Any]) -> str:
        configured = config.get("model", {}).get("kind")
        if isinstance(configured, str) and configured:
            return configured

        preferred = ["gpt_nano", "transformer", "bigram"]
        for kind in preferred:
            if kind in self.manifest.model_kinds:
                return kind
        return super().suggest_model(config)


class CVStrategy(DomainStrategy):
    """Rule-based strategy tuned for CV restoration/classification workloads."""

    def suggest_model(self, config: dict[str, Any]) -> str:
        configured = config.get("model", {}).get("kind")
        if isinstance(configured, str) and configured:
            return configured

        preferred = ["sr3", "unet", "resnet", "bicubic"]
        for kind in preferred:
            if kind in self.manifest.model_kinds:
                return kind
        return super().suggest_model(config)


class GenericStrategy(DomainStrategy):
    """Fallback strategy for domains without explicit type-specific logic."""

    pass
