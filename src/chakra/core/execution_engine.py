"""Execution strategy selection and backend execution helpers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .domain_registry import DomainManifest
from .exceptions import ConfigValidationError, ExecutionError, StrategyError


@dataclass(frozen=True)
class ExecutionDecision:
    """Result of strategy selection for a run."""

    strategy: str
    reason: str


@dataclass(frozen=True)
class SystemInfo:
    """System facts used by the strategy selector."""

    has_gpu: bool
    estimated_runtime_minutes: float | None = None
    dataset_size_mb: float | None = None


class ExecutionEngine:
    """Chooses and executes local vs Kaggle execution strategies."""

    def choose_strategy(
        self,
        config: dict[str, Any],
        system_info: SystemInfo,
        manifest: DomainManifest,
    ) -> ExecutionDecision:
        """Choose a strategy from local/kaggle/auto deterministically."""
        config = normalize_execution_config(config)
        execution_cfg = manifest.execution or {}
        lifecycle_cfg = manifest.lifecycle or {}

        requested = execution_cfg.get("default", "auto")
        supports = execution_cfg.get("supports", ["local", "kaggle"]) or ["local", "kaggle"]
        supports_set = {item for item in supports if item in {"local", "kaggle", "auto"}}

        if requested in {"local", "kaggle"}:
            if requested not in supports_set:
                raise StrategyError(f"Execution strategy '{requested}' is not supported by domain '{manifest.name}'.")
            return ExecutionDecision(strategy=requested, reason=f"manifest.execution.default={requested}")

        if requested not in {"auto", ""}:
            raise StrategyError(f"Unsupported execution default: {requested}")

        requires_gpu = bool(lifecycle_cfg.get("requires_gpu", False))
        max_runtime = lifecycle_cfg.get("max_runtime")
        runtime_threshold = _to_float(max_runtime, "lifecycle.max_runtime", strict=False)

        if requires_gpu and not system_info.has_gpu and "kaggle" in supports_set:
            return ExecutionDecision(
                strategy="kaggle",
                reason="domain requires GPU and local GPU is unavailable",
            )

        if runtime_threshold is not None and system_info.estimated_runtime_minutes is not None:
            if system_info.estimated_runtime_minutes > runtime_threshold and "kaggle" in supports_set:
                return ExecutionDecision(
                    strategy="kaggle",
                    reason=(
                        "estimated runtime exceeds lifecycle.max_runtime "
                        f"({system_info.estimated_runtime_minutes:.1f} > {runtime_threshold:.1f})"
                    ),
                )

        dataset_limit = _to_float(
            config.get("execution", {}).get("dataset_size_threshold_mb"),
            "execution.dataset_size_threshold_mb",
            strict=True,
        )
        if dataset_limit is not None and system_info.dataset_size_mb is not None:
            if system_info.dataset_size_mb > dataset_limit and "kaggle" in supports_set:
                return ExecutionDecision(
                    strategy="kaggle",
                    reason=(
                        "dataset size exceeds execution.dataset_size_threshold_mb "
                        f"({system_info.dataset_size_mb:.1f} > {dataset_limit:.1f})"
                    ),
                )

        if "local" in supports_set:
            return ExecutionDecision(strategy="local", reason="auto policy selected local execution")

        if "kaggle" in supports_set:
            return ExecutionDecision(strategy="kaggle", reason="local unsupported; fallback to kaggle")

        raise StrategyError(f"Domain '{manifest.name}' supports no known execution backends.")

    def run_local(
        self,
        cmd: list[str],
        cwd: Path,
        dry_run: bool = False,
        timeout_seconds: int | None = None,
    ) -> int:
        """Run a command locally and return process code."""
        print("$ " + " ".join(cmd))
        if dry_run:
            return 0
        try:
            completed = subprocess.run(cmd, cwd=cwd, check=False, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"Local execution timed out after {timeout_seconds}s: {' '.join(cmd)}"
            ) from exc
        return completed.returncode

    def run_kaggle(
        self,
        push_fn: Callable[[], None],
        status_fn: Callable[[], None] | None = None,
        pull_fn: Callable[[], None] | None = None,
    ) -> None:
        """Execute kaggle flow using injected side-effect functions."""
        push_fn()
        if status_fn is not None:
            status_fn()
        if pull_fn is not None:
            pull_fn()


def infer_system_info(config: dict[str, Any]) -> SystemInfo:
    """Best-effort system inspection that works without torch installed."""
    has_gpu = False
    try:
        import torch  # type: ignore

        has_gpu = bool(torch.cuda.is_available())
    except Exception:
        has_gpu = False

    normalized = normalize_execution_config(config)
    estimated_runtime = normalized.get("execution", {}).get("estimated_runtime_minutes")
    dataset_size = normalized.get("execution", {}).get("dataset_size_mb")
    return SystemInfo(
        has_gpu=has_gpu,
        estimated_runtime_minutes=estimated_runtime,
        dataset_size_mb=dataset_size,
    )


def build_train_command(module_path: str, config_path: Path, run_name: str) -> list[str]:
    """Build a deterministic python -m command for a train runner."""
    return [sys.executable, "-m", module_path, "--config", str(config_path), "--run-name", run_name]


def _to_float(value: Any, field_name: str, strict: bool) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("minutes", "").replace("minute", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            if strict:
                raise ConfigValidationError(
                    f"Invalid numeric value for '{field_name}': {value!r}"
                )
            return None
    if strict:
        raise ConfigValidationError(
            f"Invalid value type for '{field_name}': expected int|float|string, got {type(value).__name__}"
        )
    return None


def normalize_execution_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate execution config fields used by strategy selection."""
    execution = config.get("execution")
    if execution is None:
        return config
    if not isinstance(execution, dict):
        raise ConfigValidationError("Invalid value for 'execution': expected mapping/dict")

    normalized = dict(config)
    normalized_execution = dict(execution)

    normalized_execution["estimated_runtime_minutes"] = _to_float(
        normalized_execution.get("estimated_runtime_minutes"),
        "execution.estimated_runtime_minutes",
        strict=True,
    )
    normalized_execution["dataset_size_mb"] = _to_float(
        normalized_execution.get("dataset_size_mb"),
        "execution.dataset_size_mb",
        strict=True,
    )
    normalized_execution["dataset_size_threshold_mb"] = _to_float(
        normalized_execution.get("dataset_size_threshold_mb"),
        "execution.dataset_size_threshold_mb",
        strict=True,
    )
    normalized_execution["smoke_timeout_minutes"] = _to_float(
        normalized_execution.get("smoke_timeout_minutes"),
        "execution.smoke_timeout_minutes",
        strict=True,
    )

    normalized["execution"] = normalized_execution
    return normalized
