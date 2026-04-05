"""Simplified baseline models for the HNDSR scratch lane."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Classic sinusoidal timestep encoding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class ResidualTimeBlock(nn.Module):
    """Residual block conditioned on the timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, in_channels)
        out_groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    """Small conditional UNet for SR3-style smoke experiments."""

    def __init__(self, model_channels: int = 32) -> None:
        super().__init__()
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Conv2d(6, model_channels, 3, padding=1)
        self.down1 = ResidualTimeBlock(model_channels, model_channels, time_dim)
        self.down2 = nn.Conv2d(model_channels, model_channels * 2, 3, stride=2, padding=1)
        self.mid = ResidualTimeBlock(model_channels * 2, model_channels * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up2 = ResidualTimeBlock(model_channels * 2, model_channels, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, model_channels), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, 3, 3, padding=1),
        )

    def forward(self, noisy_hr: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time)
        x = torch.cat([noisy_hr, cond], dim=1)
        h0 = self.input_proj(x)
        h1 = self.down1(h0, time_emb)
        h2 = self.down2(h1)
        h3 = self.mid(h2, time_emb)
        h4 = self.up1(h3)
        h5 = self.up2(torch.cat([h4, h1], dim=1), time_emb)
        return self.out(h5)


class DDPMScheduler:
    """Minimal DDPM scheduler with deterministic reverse updates."""

    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float) -> None:
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alphas_cumprod[timesteps.cpu()].to(clean.device)
        sigma = self.sqrt_one_minus_alphas_cumprod[timesteps.cpu()].to(clean.device)
        while alpha.ndim < clean.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * clean + sigma * noise

    def step(self, predicted_noise: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prev = self.alphas_cumprod_prev[timestep].to(sample.device)
        pred_clean = (sample - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        return torch.sqrt(alpha_prev) * pred_clean + torch.sqrt(1 - alpha_prev) * predicted_noise


class SR3Baseline(nn.Module):
    """SR3-style conditional diffusion baseline for fixed 4x smoke runs."""

    def __init__(self, model_channels: int, num_timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        self.unet = ConditionalUNet(model_channels=model_channels)
        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end)

    def training_step(self, lr_upscaled: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        noise = torch.randn_like(hr)
        timesteps = self.scheduler.sample_timesteps(hr.shape[0], hr.device)
        noisy_hr = self.scheduler.add_noise(hr, noise, timesteps)
        predicted_noise = self.unet(noisy_hr, timesteps, lr_upscaled)
        loss = F.mse_loss(predicted_noise, noise)
        stats = {"timesteps_mean": float(timesteps.float().mean().item())}
        return loss, stats

    @torch.no_grad()
    def sample(self, lr_upscaled: torch.Tensor, inference_steps: int) -> torch.Tensor:
        sample = torch.randn_like(lr_upscaled)
        schedule = torch.linspace(
            self.scheduler.num_timesteps - 1,
            0,
            steps=inference_steps,
            device=lr_upscaled.device,
            dtype=torch.long,
        )
        for timestep in schedule.tolist():
            t = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
            predicted_noise = self.unet(sample, t, lr_upscaled)
            sample = self.scheduler.step(predicted_noise, int(timestep), sample)
        return sample.clamp(-1.0, 1.0)
