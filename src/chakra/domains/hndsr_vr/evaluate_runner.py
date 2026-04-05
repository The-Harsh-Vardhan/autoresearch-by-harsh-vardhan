"""Evaluate an isolated baseline and export fixed sample strips."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from chakra.core.tracker import init_tracker
from chakra.core.utils import describe_run_dirs, get_device, load_config, prepare_workspace_temp, set_seed, write_json

from .dataset import build_loaders, build_split_manifest
from .metrics import (
    bicubic_upscale,
    build_comparison_strip,
    calculate_psnr,
    calculate_ssim,
    maybe_build_lpips,
    save_grid,
    save_strip,
)
from .models import SR3Baseline


def build_model(config: dict, device: torch.device, checkpoint: str | None) -> SR3Baseline | None:
    """Instantiate and optionally load a checkpointed model."""
    if config["model"]["kind"] == "bicubic":
        return None
    if checkpoint is None:
        raise ValueError("A checkpoint is required for trainable baseline evaluation.")
    model = SR3Baseline(
        model_channels=config["model"]["model_channels"],
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def infer_sample(model: SR3Baseline | None, lr: torch.Tensor, scale: int, config: dict) -> torch.Tensor:
    """Infer a single batch for either the bicubic or SR3 baseline."""
    lr_upscaled = bicubic_upscale(lr, scale)
    if model is None:
        return lr_upscaled
    return model.sample(lr_upscaled, inference_steps=config["diffusion"]["inference_steps"])


def evaluate(config: dict, run_name: str, device: torch.device, checkpoint: str | None) -> dict[str, object]:
    """Evaluate the selected baseline and export qualitative strips."""
    dirs = describe_run_dirs(config, run_name)
    tracker = init_tracker(config, run_name, dirs["tracker"])
    bundle = build_loaders(config, seed=config["seed"])
    version = config.get("runtime", {}).get("version", run_name)

    config_manifest_path = write_json(dirs["manifests"] / "config_manifest.json", config)
    dataset_manifest_path = write_json(
        dirs["manifests"] / "dataset_split_manifest.json",
        build_split_manifest(bundle, config, version=version),
    )
    tracker.log_file_artifact(f"{run_name}-config", config_manifest_path, "config")
    tracker.log_file_artifact(f"{run_name}-dataset-manifest", dataset_manifest_path, "dataset_manifest")

    loader = bundle.val_loader
    model = build_model(config, device, checkpoint)
    lpips_fn = maybe_build_lpips(device, enabled=config["evaluation"].get("compute_lpips", True))
    strips = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    limit = config["evaluation"]["sample_limit"]
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        if batch_idx >= limit:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        scale = int(batch["scale"][0])
        name = batch["name"][0]
        with torch.no_grad():
            sr = infer_sample(model, lr, scale, config)
        strip = build_comparison_strip(lr, sr, hr, scale)
        strip_path = dirs["samples"] / f"{name}_comparison.png"
        save_strip(strip_path, strip)
        strips.append(strip)
        psnr_values.append(calculate_psnr(sr, hr))
        ssim_values.append(calculate_ssim(sr, hr))
        if lpips_fn is not None:
            lpips_values.append(float(lpips_fn(sr, hr).mean().item()))
        tracker.log_image(f"comparison/{name}", strip_path)
    grid_path = dirs["samples"] / config["evaluation"]["grid_name"]
    save_grid(grid_path, strips)
    tracker.log_file_artifact(f"{run_name}-sample-grid", grid_path, "sample_grid")
    summary = {
        "run_name": run_name,
        "version": version,
        "model_kind": config["model"]["kind"],
        "dataset_family": config["dataset"]["family"],
        "dataset_name": bundle.dataset_name,
        "pairing_mode": bundle.pairing_mode,
        "device": str(device),
        "checkpoint": checkpoint,
        "sample_grid": str(grid_path),
        "psnr_mean": float(sum(psnr_values) / max(len(psnr_values), 1)),
        "ssim_mean": float(sum(ssim_values) / max(len(ssim_values), 1)),
        "num_samples": len(psnr_values),
        "tracker_backend": tracker.backend,
        "tracker_url": tracker.run_url,
    }
    if lpips_values:
        summary["lpips_mean"] = float(sum(lpips_values) / len(lpips_values))
    tracker.log_metrics(summary)
    summary_path = write_json(dirs["metrics"] / "eval_summary.json", summary)
    tracker.log_file_artifact(f"{run_name}-eval-summary", summary_path, "metrics_summary")
    tracker.finish()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate isolated HNDSR research baseline")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--device", default=None, help="Optional torch device override")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for trainable baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_workspace_temp(config["paths"]["artifact_root"])
    set_seed(config["seed"])
    run_name = args.run_name or f"{config['project']['group']}-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    device = get_device(args.device)
    summary = evaluate(config, run_name, device, args.checkpoint)
    print(f"Evaluation complete: PSNR={summary['psnr_mean']:.2f}, SSIM={summary['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
