"""Tests for the HNDSR lifecycle review pipeline."""

import json
import pytest
from pathlib import Path
from uuid import uuid4

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _fresh_dir(name: str) -> Path:
    root = Path("artifacts") / "test-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}-{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def test_sync_review_and_mirror_pipeline(monkeypatch):
    import chakra.hndsr_vr.lifecycle as lifecycle
    import chakra.hndsr_vr.utils as utils
    from chakra.hndsr_vr.lifecycle import mirror_obsidian, next_ablation, review_run, sync_run

    # The shim lifecycle wraps core lifecycle with partial(domain='hndsr_vr')
    # For this test we need to monkeypatch the REPO_ROOT in core modules
    from chakra.core import utils as core_utils
    from chakra.core import lifecycle as core_lifecycle

    version = "vR.9"
    temp_root = _fresh_dir("lifecycle-root").resolve()
    monkeypatch.setattr(core_utils, "REPO_ROOT", temp_root)
    monkeypatch.setattr(core_lifecycle, "REPO_ROOT", temp_root)

    # The domain module imports REPO_ROOT at import time, so we must patch it there too
    from chakra.domains.hndsr_vr import lifecycle as domain_lifecycle
    monkeypatch.setattr(domain_lifecycle, "REPO_ROOT", temp_root)

    benchmark_dir = temp_root / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "hndsr_vr_registry.json").write_text(
        json.dumps(
            {
                "lane": "HNDSR scratch",
                "control_baselines": {
                    "kaggle_4x_bicubic_smoke": {
                        "dataset": "kaggle_4x",
                        "protocol": "paired 4x smoke pack",
                        "psnr_mean": 30.6039,
                        "ssim_mean": 0.7365,
                        "source": "test fixture",
                    }
                },
                "paper_targets": [],
            }
        ),
        encoding="utf-8",
    )

    source_dir = temp_root / "incoming" / "kaggle-output"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"{version}_HNDSR.ipynb").write_text("{}", encoding="utf-8")
    (source_dir / f"{version}_train_best.pt").write_text("checkpoint", encoding="utf-8")
    (source_dir / f"{version}_train_grid.png").write_text("grid", encoding="utf-8")
    (source_dir / "train_summary.json").write_text(json.dumps({"best_val_loss": 0.5}), encoding="utf-8")
    (source_dir / "eval_summary.json").write_text(
        json.dumps({"psnr_mean": 31.2, "ssim_mean": 0.78}),
        encoding="utf-8",
    )

    sync_run(version, source_dir=str(source_dir), wandb_url="https://wandb.example/run/vr9")
    review_run(version)
    mirror_root = temp_root / "obsidian"
    mirror_obsidian(version, output_dir=str(mirror_root))
    next_ablation(version)

    manifest = json.loads((temp_root / f"artifacts/runs/{version}/run_manifest.json").read_text(encoding="utf-8"))
    review_payload = json.loads((temp_root / f"reports/reviews/{version}_HNDSR.review.json").read_text(encoding="utf-8"))
    mirror_note = (mirror_root / f"{version}_HNDSR.md").read_text(encoding="utf-8")
    ablation_note = (temp_root / "reports" / "generated" / f"{version}_HNDSR.next_ablation.md").read_text(encoding="utf-8")

    assert manifest["wandb_url"] == "https://wandb.example/run/vr9"
    assert review_payload["decision"] == "freeze and fork next version"
    assert f"{version}.1:" in mirror_note or f"{version}" in mirror_note
    assert f"{version}" in ablation_note
