"""Tests for the domain registry and manifest loading."""

from autoresearch_hv.core.domain_registry import DomainManifest, discover_domains, get_domain


def test_discover_finds_both_domains():
    domains = discover_domains()
    assert "hndsr_vr" in domains
    assert "nlp_lm" in domains


def test_hndsr_manifest_fields():
    manifest = get_domain("hndsr_vr")
    assert manifest.display_name == "HNDSR Satellite Super-Resolution"
    assert manifest.primary_metric == "psnr_mean"
    assert manifest.metric_direction == "higher_is_better"
    assert "sr3" in manifest.model_kinds
    assert "bicubic" in manifest.model_kinds
    assert manifest.entrypoints["lifecycle"] == "autoresearch_hv.domains.hndsr_vr.lifecycle"


def test_nlp_manifest_fields():
    manifest = get_domain("nlp_lm")
    assert manifest.display_name == "NLP Language Modelling"
    assert manifest.primary_metric == "val_bpb"
    assert manifest.metric_direction == "lower_is_better"
    assert "gpt_nano" in manifest.model_kinds
    assert "bigram" in manifest.model_kinds
    assert manifest.entrypoints["lifecycle"] == "autoresearch_hv.domains.nlp_lm.lifecycle"


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
