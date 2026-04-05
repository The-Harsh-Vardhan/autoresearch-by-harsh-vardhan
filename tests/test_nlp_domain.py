"""Tests for the NLP language-modelling domain."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def test_gpt_nano_forward():
    from autoresearch_hv.domains.nlp_lm.models import GPTNano
    model = GPTNano(vocab_size=26, seq_len=32, n_embd=32, n_head=2, n_layer=2)
    x = torch.randint(0, 26, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 26)


def test_bigram_forward():
    from autoresearch_hv.domains.nlp_lm.models import BigramBaseline
    model = BigramBaseline(vocab_size=26)
    x = torch.randint(0, 26, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 26)


def test_char_dataset():
    from autoresearch_hv.domains.nlp_lm.dataset import CharTextDataset
    text = "hello world this is a test of the char dataset module"
    ds = CharTextDataset(text, seq_len=8)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["input_ids"].shape == (8,)
    assert sample["target_ids"].shape == (8,)
    assert ds.vocab_size > 0


def test_metrics_shapes():
    from autoresearch_hv.domains.nlp_lm.metrics import calculate_bpb, calculate_cross_entropy, calculate_perplexity
    logits = torch.randn(2, 16, 26)
    targets = torch.randint(0, 26, (2, 16))
    ce = calculate_cross_entropy(logits, targets)
    ppl = calculate_perplexity(logits, targets)
    bpb = calculate_bpb(logits, targets)
    assert isinstance(ce, float)
    assert isinstance(ppl, float)
    assert isinstance(bpb, float)
    assert ppl > 0
    assert bpb > 0


def test_nlp_version_helpers():
    # This doesn't actually need torch, but keep it grouped with NLP tests
    from autoresearch_hv.domains.nlp_lm.utils import version_stem, version_slug
    assert version_stem("v1.0") == "v1.0_NLP_LM"
    assert version_slug("v1.0") == "v10-nlp-lm"


def test_nlp_lifecycle_hooks_resolve_paths():
    from autoresearch_hv.domains.nlp_lm.lifecycle import LifecycleHooks
    hooks = LifecycleHooks()
    paths = hooks.resolve_version_paths("v1.0")
    assert "v1.0_NLP_LM" in str(paths.notebook)
    assert "nlp_lm" in str(paths.configs["train"])
