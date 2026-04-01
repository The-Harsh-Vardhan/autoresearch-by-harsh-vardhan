import json
from pathlib import Path
from uuid import uuid4

from autoresearch_hv.hndsr_vr.lifecycle import mirror_obsidian, next_ablation, review_run, sync_run
from autoresearch_hv.hndsr_vr.utils import read_json


def _fresh_dir(name: str) -> Path:
    root = Path("artifacts") / "test-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name}-{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def test_sync_review_and_mirror_pipeline():
    version = "vR.1"
    source_dir = _fresh_dir("kaggle-output")
    (source_dir / "vR.1_HNDSR.ipynb").write_text("{}", encoding="utf-8")
    (source_dir / "vR.1_train_best.pt").write_text("checkpoint", encoding="utf-8")
    (source_dir / "vR.1_train_grid.png").write_text("grid", encoding="utf-8")
    (source_dir / "train_summary.json").write_text(json.dumps({"best_val_loss": 0.5}), encoding="utf-8")
    (source_dir / "eval_summary.json").write_text(
        json.dumps({"psnr_mean": 31.2, "ssim_mean": 0.78}),
        encoding="utf-8",
    )

    sync_run(version, source_dir=str(source_dir), wandb_url="https://wandb.example/run/vr1")
    review_run(version)
    mirror_root = _fresh_dir("obsidian")
    mirror_obsidian(version, output_dir=str(mirror_root))
    next_ablation(version)

    manifest = read_json("artifacts/runs/vR.1/run_manifest.json")
    review_payload = read_json("reports/reviews/vR.1_HNDSR.review.json")
    mirror_note = (mirror_root / "vR.1_HNDSR.md").read_text(encoding="utf-8")
    ablation_note = Path("reports/generated/vR.1_HNDSR.next_ablation.md").read_text(encoding="utf-8")

    assert manifest["wandb_url"] == "https://wandb.example/run/vr1"
    assert review_payload["decision"] == "freeze and fork next version"
    assert "vR.1.1:" in mirror_note
    assert "vR.1.1:" in ablation_note

