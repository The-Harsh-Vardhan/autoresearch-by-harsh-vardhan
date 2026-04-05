"""Tests for the HNDSR runtime contracts."""

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

from pathlib import Path
from uuid import uuid4


def _write_fake_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (96, 96)) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=color)
    image.save(path)


def _fresh_dir(name: str) -> Path:
    base = Path("artifacts") / "test-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    root = base / f"{name}-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_version_helpers():
    from chakra.hndsr_vr.utils import version_stem, version_slug
    assert version_stem("vR.1") == "vR.1_HNDSR"
    assert version_slug("vR.1") == "vr1-hndsr"


def test_sr3_forward_loss_contract():
    from chakra.hndsr_vr.models import SR3Baseline
    model = SR3Baseline(model_channels=16, num_timesteps=32, beta_start=1.0e-4, beta_end=0.02)
    lr_upscaled = torch.randn(2, 3, 64, 64)
    hr = torch.randn(2, 3, 64, 64)
    loss, stats = model.training_step(lr_upscaled, hr)
    assert loss.ndim == 0
    assert "timesteps_mean" in stats


def test_null_tracker_accepts_logs():
    from chakra.hndsr_vr.tracker import NullTracker
    tracker = NullTracker(run_dir="artifacts/.tmp/test-null-tracker")
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.log_text("status", "ok")
    tracker.finish()


def test_synthetic_dataset_generates_deterministic_4x_pair():
    from chakra.hndsr_vr.dataset import SyntheticSatellitePairDataset
    root = _fresh_dir("synthetic-ucmerced")
    _write_fake_image(root / "airport" / "sample_001.png", (200, 100, 50))
    dataset = SyntheticSatellitePairDataset(str(root), patch_size=64, training=False, scale_factor=4)
    sample_a = dataset[0]
    sample_b = dataset[0]
    assert sample_a["name"] == "airport__sample_001"
    assert sample_a["scale"] == 4
    assert tuple(sample_a["hr"].shape) == (3, 64, 64)
    assert tuple(sample_a["lr"].shape) == (3, 16, 16)
    assert torch.allclose(sample_a["hr"], sample_b["hr"])
    assert torch.allclose(sample_a["lr"], sample_b["lr"])


def test_kaggle_pairing_lane_preserves_traceable_names():
    from chakra.hndsr_vr.dataset import build_loaders
    from chakra.hndsr_vr.utils import load_config
    root = _fresh_dir("kaggle-paired")
    hr_root = root / "kaggle" / "HR"
    lr_root = root / "kaggle" / "LR"
    _write_fake_image(hr_root / "tile_001.png", (220, 40, 40), size=(96, 96))
    _write_fake_image(lr_root / "tile_001.png", (200, 30, 30), size=(24, 24))
    _write_fake_image(hr_root / "tile_002.png", (120, 80, 40), size=(96, 96))
    _write_fake_image(lr_root / "tile_002.png", (110, 70, 30), size=(24, 24))
    config = load_config("configs/hndsr_vr/base.yaml")
    config["dataset"] = {"family": "kaggle", "name": "kaggle_4x", "pairing_mode": "paired", "scale_factor": 4}
    config["paths"]["datasets"]["kaggle_4x"]["hr_dir"] = str(hr_root)
    config["paths"]["datasets"]["kaggle_4x"]["lr_dir"] = str(lr_root)
    config["data"]["val_split"] = 0.5
    config["data"]["train_limit"] = None
    config["data"]["val_limit"] = None
    bundle = build_loaders(config, seed=42)
    first_batch = next(iter(bundle.val_loader))
    assert first_batch["name"][0] in {"tile_001", "tile_002"}
    assert int(first_batch["scale"][0]) == 4


def test_rendered_notebook_contains_required_contract_fragments():
    from chakra.hndsr_vr.lifecycle import resolve_version_paths, render_notebook
    paths = resolve_version_paths("vR.9")
    notebook = render_notebook("vR.9", parent="vR.8", paths=paths)
    assert "## Weights & Biases Setup" in notebook
    assert "python -m chakra validate-version --version vR.9" in notebook
    assert "python -m chakra.hndsr_vr.train_runner --config" in notebook
