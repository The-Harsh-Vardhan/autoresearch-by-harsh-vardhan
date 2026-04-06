"""Tests for execution strategy selection and backend safety gates."""

from pathlib import Path
import sys

from chakra.core.domain_registry import get_domain
from chakra.core.exceptions import ConfigValidationError, ExecutionError
from chakra.core.execution_engine import ExecutionEngine, SystemInfo


def test_auto_selects_kaggle_when_gpu_required_and_missing() -> None:
    manifest = get_domain("nlp_lm")
    engine = ExecutionEngine()
    decision = engine.choose_strategy(
        config={"execution": {"estimated_runtime_minutes": 20}},
        system_info=SystemInfo(has_gpu=False, estimated_runtime_minutes=20, dataset_size_mb=100),
        manifest=manifest,
    )
    assert decision.strategy == "kaggle"


def test_auto_selects_local_when_gpu_not_required() -> None:
    manifest = get_domain("tabular_cls")
    engine = ExecutionEngine()
    decision = engine.choose_strategy(
        config={"execution": {"estimated_runtime_minutes": 5}},
        system_info=SystemInfo(has_gpu=False, estimated_runtime_minutes=5, dataset_size_mb=5),
        manifest=manifest,
    )
    assert decision.strategy == "local"


def test_auto_selects_kaggle_on_runtime_threshold() -> None:
    manifest = get_domain("nlp_lm")
    engine = ExecutionEngine()
    decision = engine.choose_strategy(
        config={"execution": {"estimated_runtime_minutes": 120}},
        system_info=SystemInfo(has_gpu=True, estimated_runtime_minutes=120, dataset_size_mb=20),
        manifest=manifest,
    )
    assert decision.strategy == "kaggle"


def test_explicit_default_is_respected() -> None:
    manifest = get_domain("tabular_cls")
    engine = ExecutionEngine()
    decision = engine.choose_strategy(
        config={},
        system_info=SystemInfo(has_gpu=False),
        manifest=manifest,
    )
    assert decision.strategy == "local"


def test_invalid_execution_numeric_raises_config_error() -> None:
    manifest = get_domain("nlp_lm")
    engine = ExecutionEngine()
    try:
        engine.choose_strategy(
            config={"execution": {"estimated_runtime_minutes": "not-a-number"}},
            system_info=SystemInfo(has_gpu=False),
            manifest=manifest,
        )
        assert False, "Expected ConfigValidationError"
    except ConfigValidationError as exc:
        assert "execution.estimated_runtime_minutes" in str(exc)


def test_run_local_timeout_raises_execution_error() -> None:
    engine = ExecutionEngine()
    cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    try:
        engine.run_local(cmd=cmd, cwd=Path.cwd(), timeout_seconds=1)
        assert False, "Expected ExecutionError"
    except ExecutionError as exc:
        assert "timed out" in str(exc)
