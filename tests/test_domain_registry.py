"""Tests for the domain registry and manifest loading."""

from chakra.core.domain_registry import DomainManifest, discover_domains, get_domain


def test_discover_finds_both_domains():
    domains = discover_domains()
    assert "hndsr_vr" in domains
    assert "nlp_lm" in domains
    assert "tabular_cls" in domains


def test_hndsr_manifest_fields():
    manifest = get_domain("hndsr_vr")
    assert manifest.display_name == "HNDSR Satellite Super-Resolution"
    assert manifest.primary_metric == "psnr_mean"
    assert manifest.metric_direction == "higher_is_better"
    assert "sr3" in manifest.model_kinds
    assert "bicubic" in manifest.model_kinds
    assert manifest.entrypoints["lifecycle"] == "chakra.domains.hndsr_vr.lifecycle"


def test_nlp_manifest_fields():
    manifest = get_domain("nlp_lm")
    assert manifest.display_name == "NLP Language Modelling"
    assert manifest.domain == "nlp"
    assert manifest.task == "generation"
    assert manifest.primary_metric == "val_bpb"
    assert manifest.metric_direction == "lower_is_better"
    assert "gpt_nano" in manifest.model_kinds
    assert "bigram" in manifest.model_kinds
    assert manifest.execution["default"] == "auto"
    assert manifest.entrypoints["lifecycle"] == "chakra.domains.nlp_lm.lifecycle"


def test_unknown_domain_raises_key_error():
    try:
        get_domain("nonexistent_domain")
        assert False, "Expected KeyError"
    except KeyError as exc:
        assert "nonexistent_domain" in str(exc)


def test_manifest_from_dict():
    data = {
        "name": "test_domain",
        "display_name": "Test Domain",
        "version_pattern": r"^v\d+$",
        "model_kinds": ["baseline"],
        "primary_metric": "accuracy",
        "metric_direction": "higher_is_better",
        "benchmark_registry": "benchmarks/test.json",
        "config_dir": "configs/test",
        "programs_doc": "programs/test.md",
        "entrypoints": {"lifecycle": "test.lifecycle"},
    }
    manifest = DomainManifest.from_dict(data)
    assert manifest.name == "test_domain"
    assert manifest.primary_metric == "accuracy"
    assert manifest.model_kinds == ["baseline"]


def test_manifest_v2_optional_fields_are_parsed():
    data = {
        "name": "x_tab",
        "display_name": "X Tab",
        "version_pattern": r"^v\\d+$",
        "model_kinds": ["mlp"],
        "primary_metric": "acc",
        "metric_direction": "higher_is_better",
        "benchmark_registry": "benchmarks/x.json",
        "config_dir": "configs/x",
        "programs_doc": "programs/x.md",
        "entrypoints": {"lifecycle": "x.lifecycle", "train_runner": "x.train"},
        "domain": "tb",
        "task": "classification",
        "meta": {"modality": "tabular", "supervision": "supervised", "output_type": "discriminative"},
        "capabilities": {"supports_generation": False},
        "defaults": {"model_family": "mlp"},
        "metrics": {"primary": "acc", "secondary": ["f1"]},
        "search_space": {"architectures": ["mlp"], "optimizers": ["adamw"], "schedulers": ["none"]},
        "failure_modes": ["overfit"],
        "ablation_templates": {"architecture": ["increase_hidden_dim"]},
        "lifecycle": {"requires_gpu": False, "supports_local": True, "max_runtime": 10},
        "execution": {"default": "local", "supports": ["local", "kaggle"]},
        "intent": "benchmark",
        "agents": {"planner": "enabled"},
    }
    manifest = DomainManifest.from_dict(data)
    assert manifest.domain == "tb"
    assert manifest.metrics["secondary"] == ["f1"]
    assert manifest.execution["default"] == "local"


def test_manifest_v2_invalid_execution_default_raises():
    data = {
        "name": "x_bad",
        "display_name": "X Bad",
        "version_pattern": r"^v\\d+$",
        "model_kinds": ["baseline"],
        "primary_metric": "acc",
        "metric_direction": "higher_is_better",
        "benchmark_registry": "benchmarks/x.json",
        "config_dir": "configs/x",
        "programs_doc": "programs/x.md",
        "entrypoints": {"lifecycle": "x.lifecycle"},
        "execution": {"default": "cluster", "supports": ["local"]},
    }
    try:
        DomainManifest.from_dict(data)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "execution.default" in str(exc)
