"""Tests for deterministic domain strategy behavior."""

from chakra.core.domain_registry import DomainManifest, get_domain, load_domain_strategy


def test_load_domain_strategy_picks_nlp_strategy() -> None:
    manifest = get_domain("nlp_lm")
    strategy = load_domain_strategy(manifest)
    assert strategy.__class__.__name__ == "NLPStrategy"


def test_load_domain_strategy_picks_cv_strategy() -> None:
    manifest = get_domain("hndsr_vr")
    strategy = load_domain_strategy(manifest)
    assert strategy.__class__.__name__ == "CVStrategy"


def test_generate_ablation_plan_uses_templates() -> None:
    manifest = get_domain("tabular_cls")
    strategy = load_domain_strategy(manifest)
    plan = strategy.generate_ablation_plan({"runtime": {"version": "v1.2"}})
    assert plan
    assert plan[0]["id"].startswith("v1.2:")
    assert {item["axis"] for item in plan}.issubset({"architecture", "optimization", "data"})


def test_detect_failure_modes_matches_declared_keywords() -> None:
    manifest = DomainManifest.from_dict(
        {
            "name": "x_test",
            "display_name": "X Test",
            "version_pattern": r"^v\\d+$",
            "model_kinds": ["baseline"],
            "primary_metric": "score",
            "metric_direction": "higher_is_better",
            "benchmark_registry": "benchmarks/x.json",
            "config_dir": "configs/x",
            "programs_doc": "programs/x.md",
            "entrypoints": {"lifecycle": "x.lifecycle"},
            "failure_modes": ["oom", "nan_loss"],
        }
    )
    strategy = load_domain_strategy(manifest)
    detected = strategy.detect_failure_modes({"stderr": "Run aborted due to OOM at step 200"})
    assert detected == ["oom"]
