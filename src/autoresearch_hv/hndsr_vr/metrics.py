"""Metrics and visualization helpers for the research track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid, save_image


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map tensors from [-1, 1] back to [0, 1]."""
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def calculate_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute mean PSNR over a batch."""
    sr_np = denormalize(sr).detach().cpu().numpy()
    hr_np = denormalize(hr).detach().cpu().numpy()
    values = [
        psnr(sr_np[i].transpose(1, 2, 0), hr_np[i].transpose(1, 2, 0), data_range=1.0)
        for i in range(sr_np.shape[0])
    ]
    return float(np.mean(values))


def calculate_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute mean SSIM over a batch."""
    sr_np = denormalize(sr).detach().cpu().numpy()
    hr_np = denormalize(hr).detach().cpu().numpy()
    values = [
        ssim(sr_np[i].transpose(1, 2, 0), hr_np[i].transpose(1, 2, 0), data_range=1.0, channel_axis=2)
        for i in range(sr_np.shape[0])
    ]
    return float(np.mean(values))


def maybe_build_lpips(device: torch.device, enabled: bool) -> torch.nn.Module | None:
    """Create an LPIPS scorer if the dependency is available."""
    if not enabled:
        return None
    try:
        import lpips
    except Exception:
        return None
    return lpips.LPIPS(net="alex").to(device)


def bicubic_upscale(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Upscale LR input using bicubic interpolation."""
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def build_comparison_strip(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> torch.Tensor:
    """Create a [LR-nearest | bicubic | SR | HR] strip for one sample."""
    lr_nearest = F.interpolate(lr, scale_factor=scale, mode="nearest")
    bicubic = bicubic_upscale(lr, scale)
    images = torch.cat(
        [
            denormalize(lr_nearest),
            denormalize(bicubic),
            denormalize(sr),
            denormalize(hr),
        ],
        dim=0,
    )
    return make_grid(images, nrow=4, padding=2)


def save_strip(path: str | Path, strip: torch.Tensor) -> None:
    """Persist a comparison strip."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    save_image(strip, resolved)


def save_grid(path: str | Path, strips: list[torch.Tensor]) -> None:
    """Persist a montage of comparison strips."""
    if not strips:
        return
    grid = make_grid(strips, nrow=1, padding=6)
    save_strip(path, grid)
