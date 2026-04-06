"""End-to-end tests for run_execution orchestration."""

from chakra.core import lifecycle
from chakra.core.exceptions import ExecutionError


def test_run_execution_local_dry_run_emits_strategy(capsys):
    lifecycle.run_execution("tabular_cls", "v1.0", strategy="local", dry_run=True)
    output = capsys.readouterr().out
    assert "Resolved train config:" in output
    assert "Requested execution strategy: local" in output
    assert "Execution decision: local" in output


def test_run_execution_kaggle_dry_run_blocks_on_smoke_failure(monkeypatch):
    def _fail_smoke(*_args, **_kwargs):
        return False

    monkeypatch.setattr(lifecycle, "_run_local_smoke_gate", _fail_smoke)

    try:
        lifecycle.run_execution("nlp_lm", "v1.0", strategy="kaggle", dry_run=True)
        assert False, "Expected ExecutionError"
    except ExecutionError as exc:
        assert "smoke gate failed" in str(exc).lower()


def test_run_execution_kaggle_dry_run_proceeds_when_smoke_passes(monkeypatch, capsys):
    calls: list[str] = []

    def _pass_smoke(*_args, **_kwargs):
        calls.append("smoke")
        return True

    def _fake_run_kaggle(push_fn, status_fn=None, pull_fn=None):
        calls.append("kaggle")
        push_fn()
        if status_fn is not None:
            status_fn()
        if pull_fn is not None:
            pull_fn()

    monkeypatch.setattr(lifecycle, "_run_local_smoke_gate", _pass_smoke)
    monkeypatch.setattr(lifecycle.ExecutionEngine, "run_kaggle", lambda self, push_fn, status_fn=None, pull_fn=None: _fake_run_kaggle(push_fn, status_fn, pull_fn))

    lifecycle.run_execution("nlp_lm", "v1.0", strategy="kaggle", dry_run=True, pull_outputs=True)
    output = capsys.readouterr().out
    assert "Execution decision: kaggle" in output
    assert calls[0] == "smoke"
    assert "kaggle" in calls
