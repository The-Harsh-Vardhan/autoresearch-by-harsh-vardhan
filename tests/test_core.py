"""Tests for core modules that do NOT require torch/pyyaml."""

import json
import os
from pathlib import Path
from uuid import uuid4

from autoresearch_hv.core.utils import (
    _parse_simple_yaml,
    _yaml_scalar,
    flatten_config,
    get_device,
    load_yaml_text,
    set_seed,
)
from autoresearch_hv.core.tracker import NullTracker


# ---------------------------------------------------------------------------
# YAML parser tests
# ---------------------------------------------------------------------------


def test_yaml_scalar_types():
    assert _yaml_scalar("true") is True
    assert _yaml_scalar("false") is False
    assert _yaml_scalar("null") is None
    assert _yaml_scalar("42") == 42
    assert _yaml_scalar("3.14") == 3.14
    assert _yaml_scalar('"quoted"') == "quoted"
    assert _yaml_scalar("plain_string") == "plain_string"


def test_parse_simple_yaml_flat():
    text = "name: my_domain\nversion: 1\nenabled: true\n"
    result = _parse_simple_yaml(text)
    assert result == {"name": "my_domain", "version": 1, "enabled": True}


def test_parse_simple_yaml_nested_dict():
    text = "entrypoints:\n  lifecycle: my.module\n  runner: my.runner\n"
    result = _parse_simple_yaml(text)
    assert result == {"entrypoints": {"lifecycle": "my.module", "runner": "my.runner"}}


def test_parse_simple_yaml_dash_list():
    text = "model_kinds:\n  - sr3\n  - bicubic\n"
    result = _parse_simple_yaml(text)
    assert result == {"model_kinds": ["sr3", "bicubic"]}


def test_parse_simple_yaml_inline_list():
    text = "tags: [phase1, nlp, v1.0]\n"
    result = _parse_simple_yaml(text)
    assert result == {"tags": ["phase1", "nlp", "v1.0"]}


def test_parse_simple_yaml_comments_and_blanks():
    text = "# This is a comment\n\nname: test\n# Another comment\nvalue: 42\n"
    result = _parse_simple_yaml(text)
    assert result == {"name": "test", "value": 42}


def test_parse_simple_yaml_full_domain_manifest():
    """Parse a realistic domain.yaml without pyyaml."""
    text = """name: test_domain
display_name: Test Domain
version_pattern: "^v\\\\d+$"
model_kinds:
  - baseline
  - advanced
primary_metric: accuracy
metric_direction: higher_is_better
config_dir: configs/test
entrypoints:
  lifecycle: test.lifecycle
  runner: test.runner
"""
    result = _parse_simple_yaml(text)
    assert result["name"] == "test_domain"
    assert result["model_kinds"] == ["baseline", "advanced"]
    assert result["entrypoints"]["lifecycle"] == "test.lifecycle"
    assert result["primary_metric"] == "accuracy"


def test_load_yaml_text_uses_fallback():
    """load_yaml_text should work even if pyyaml is available."""
    result = load_yaml_text("key: value\ncount: 3\n")
    assert result["key"] == "value"
    assert result["count"] == 3


# ---------------------------------------------------------------------------
# Config utility tests
# ---------------------------------------------------------------------------


def test_flatten_config():
    config = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_config(config)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flatten_config_empty():
    assert flatten_config({}) == {}


# ---------------------------------------------------------------------------
# Device and seed tests
# ---------------------------------------------------------------------------


def test_get_device_without_torch():
    """get_device should return a string when torch is not available."""
    result = get_device()
    # Returns either a torch.device or "cpu" string
    assert str(result) in {"cpu", "cuda", "mps"} or hasattr(result, "type")


def test_get_device_explicit():
    result = get_device("cpu")
    assert str(result) == "cpu"


def test_set_seed_does_not_crash():
    """set_seed should work regardless of torch/numpy availability."""
    set_seed(42)


# ---------------------------------------------------------------------------
# NullTracker tests
# ---------------------------------------------------------------------------


def _fresh_tracker_dir() -> Path:
    base = Path("artifacts") / "test-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"tracker-{uuid4().hex[:8]}"


def test_null_tracker_records_metrics():
    tracker = NullTracker(_fresh_tracker_dir())
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
    tracker.log_metrics({"loss": 0.3, "accuracy": 0.95}, step=2)
    tracker.log_text("status", "completed")
    tracker.finish()

    records_path = tracker.run_dir / "tracker_records.json"
    assert records_path.exists()
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "null"
    assert len(payload["records"]) == 3
    assert payload["summary"]["status"] == "completed"


def test_null_tracker_logs_artifacts():
    tracker = NullTracker(_fresh_tracker_dir())
    tracker.log_file_artifact("test-file", "path/to/file.pt", "checkpoint", {"version": "v1"})
    tracker.log_dir_artifact("test-dir", "path/to/dir/", "dataset")
    tracker.finish()

    payload = json.loads((tracker.run_dir / "tracker_records.json").read_text(encoding="utf-8"))
    file_records = [r for r in payload["records"] if r["type"] == "artifact_file"]
    dir_records = [r for r in payload["records"] if r["type"] == "artifact_dir"]
    assert len(file_records) == 1
    assert file_records[0]["metadata"] == {"version": "v1"}
    assert len(dir_records) == 1


# ---------------------------------------------------------------------------
# Version label tests
# ---------------------------------------------------------------------------


def test_next_version_labels():
    from autoresearch_hv.core.lifecycle import _next_version_labels

    # HNDSR-style: vR.1 → vR.2, vR.1.1
    major, minor = _next_version_labels("vR.1")
    assert major == "vR.2"
    assert minor == "vR.1.1"

    # NLP-style: v1.0 → v1.1, v1.0.1
    major, minor = _next_version_labels("v1.0")
    assert major == "v1.1"
    assert minor == "v1.0.1"

    # Three-part: v1.2.3 → v1.2.4, v1.2.3.1
    major, minor = _next_version_labels("v1.2.3")
    assert major == "v1.2.4"
    assert minor == "v1.2.3.1"
