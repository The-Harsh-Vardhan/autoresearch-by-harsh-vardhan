"""Tests for lifecycle integration with execution orchestration."""

import pytest

from chakra.core import lifecycle
from chakra.core.exceptions import ExecutionError


def test_run_execution_blocks_kaggle_when_smoke_fails(monkeypatch):
    pushed = {"value": False}

    def _fake_smoke(*_args, **_kwargs):
        return False

    def _fake_push(*_args, **_kwargs):
        pushed["value"] = True

    monkeypatch.setattr(lifecycle, "_run_local_smoke_gate", _fake_smoke)
    monkeypatch.setattr(lifecycle, "push_kaggle", _fake_push)

    with pytest.raises(ExecutionError) as exc:
        lifecycle.run_execution("nlp_lm", "v1.0", strategy="kaggle", dry_run=True)

    assert "Local smoke gate failed" in str(exc.value)
    assert pushed["value"] is False


def test_run_execution_local_dry_run_succeeds():
    # Dry-run local execution should not require torch and should not raise.
    lifecycle.run_execution("tabular_cls", "v1.0", strategy="local", dry_run=True)


def test_run_execution_smoke_gate_timeout_is_reported(monkeypatch):
    def _fake_smoke(*_args, **_kwargs):
        raise ExecutionError("Local execution timed out after 1s: python -m train")

    monkeypatch.setattr(lifecycle, "_run_local_smoke_gate", _fake_smoke)

    with pytest.raises(ExecutionError) as exc:
        lifecycle.run_execution("nlp_lm", "v1.0", strategy="kaggle", dry_run=True)

    assert "timed out" in str(exc.value)
