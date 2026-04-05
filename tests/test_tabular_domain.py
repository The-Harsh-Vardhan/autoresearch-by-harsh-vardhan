"""Tests for the tabular_cls domain."""

from __future__ import annotations

import pytest


def test_tabular_cls_discovered():
    """The tabular_cls domain is auto-discovered by the registry."""
    from chakra.core.domain_registry import discover_domains
    domains = discover_domains()
    assert "tabular_cls" in domains
    manifest = domains["tabular_cls"]
    assert manifest.display_name == "Tabular Classification"
    assert manifest.primary_metric == "accuracy"
    assert manifest.metric_direction == "higher_is_better"
    assert "logistic" in manifest.model_kinds
    assert "mlp" in manifest.model_kinds


def test_tabular_cls_lifecycle_hooks_protocol():
    """LifecycleHooks satisfies the DomainLifecycleHooks protocol."""
    from chakra.core.interfaces import DomainLifecycleHooks
    from chakra.domains.tabular_cls.lifecycle import LifecycleHooks
    hooks = LifecycleHooks()
    assert isinstance(hooks, DomainLifecycleHooks)


def test_tabular_version_naming():
    """version_stem and version_slug produce expected outputs."""
    from chakra.domains.tabular_cls.utils import version_slug, version_stem
    assert version_stem("v1.0") == "v1.0_Tabular_CLS"
    assert version_slug("v1.0") == "v1-0-tabular-cls"


def test_tabular_resolve_paths():
    """resolve_version_paths returns a VersionPaths with correct structure."""
    from chakra.domains.tabular_cls.lifecycle import LifecycleHooks
    hooks = LifecycleHooks()
    paths = hooks.resolve_version_paths("v1.0")
    assert paths.version == "v1.0"
    assert "v1.0_Tabular_CLS" in paths.notebook.name
    assert "control" in paths.configs
    assert "smoke" in paths.configs
    assert "train" in paths.configs


def test_tabular_build_configs():
    """build_version_configs generates control/smoke/train variants."""
    from chakra.domains.tabular_cls.lifecycle import LifecycleHooks
    hooks = LifecycleHooks()
    configs = hooks.build_version_configs("v1.0", parent=None, lineage="scratch")
    assert set(configs.keys()) == {"control", "smoke", "train"}
    assert configs["control"]["model"]["kind"] == "logistic"
    assert configs["smoke"]["model"]["kind"] == "mlp"
    assert configs["train"]["model"]["kind"] == "mlp"
    assert configs["control"]["training"]["epochs"] == 1
    assert configs["smoke"]["training"]["epochs"] == 3
    assert configs["train"]["training"]["epochs"] == 30


def test_tabular_iris_dataset():
    """Iris dataset loads correctly with expected shape."""
    from chakra.domains.tabular_cls.dataset import build_loaders
    config = {
        "data": {"dataset": "iris", "val_split": 0.2, "batch_size": 32},
    }
    bundle = build_loaders(config, seed=42)
    assert bundle.dataset_name == "iris"
    assert bundle.num_features == 4
    assert bundle.num_classes == 3
    assert bundle.train_size + bundle.val_size == 150


def test_tabular_models_forward():
    """Both models produce correct output shapes."""
    import torch
    from chakra.domains.tabular_cls.models import LogisticBaseline, SmallMLP

    x = torch.randn(8, 4)
    logistic = LogisticBaseline(num_features=4, num_classes=3)
    out = logistic(x)
    assert out.shape == (8, 3)

    mlp = SmallMLP(num_features=4, num_classes=3, hidden_dim=32, dropout=0.1)
    out = mlp(x)
    assert out.shape == (8, 3)


def test_tabular_metrics():
    """Metric functions return sensible values."""
    import torch
    from chakra.domains.tabular_cls.metrics import calculate_accuracy, calculate_cross_entropy, calculate_f1

    logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
    targets = torch.tensor([0, 1, 2])
    assert calculate_accuracy(logits, targets) == pytest.approx(1.0)
    assert calculate_cross_entropy(logits, targets) > 0
    assert 0.0 < calculate_f1(logits, targets, 3) <= 1.0


def test_tabular_roast_lines():
    """roast_lines returns non-empty list."""
    from chakra.domains.tabular_cls.lifecycle import LifecycleHooks
    hooks = LifecycleHooks()
    lines = hooks.roast_lines()
    assert len(lines) >= 1
    assert all(isinstance(line, str) for line in lines)
