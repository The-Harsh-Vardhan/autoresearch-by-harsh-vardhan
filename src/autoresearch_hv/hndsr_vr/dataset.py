"""Dataset helpers for the HNDSR scratch lane."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_EXTENSIONS = (
    "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
    "*.PNG", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF",
)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for paired loaders and dataset metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int
    dataset_name: str
    pairing_mode: str
    train_manifest: list[str]
    val_manifest: list[str]


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _collect_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(root.rglob(pattern))
    return sorted(images)


def _trace_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return "__".join(relative.with_suffix("").parts)


def _limited_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def _split_indices(size: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    if size < 2:
        raise ValueError("Need at least two samples to create deterministic train/val splits.")
    val_size = max(1, int(round(size * val_split)))
    train_size = size - val_size
    if train_size < 1:
        train_size = 1
        val_size = size - train_size
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    return order[:train_size], order[train_size:]


def _resolve_spec(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    registry = config["paths"]["datasets"]
    dataset_name = dataset_cfg["name"]
    if dataset_name not in registry:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Add it under paths.datasets.")
    spec = dict(registry[dataset_name])
    spec.update(dataset_cfg)
    return spec


def _names_from_subset(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
    else:
        base = dataset
        indices = list(range(len(dataset)))
    names: list[str] = []
    if hasattr(base, "pairs"):
        names = [str(base.pairs[index][0]) for index in indices]
    elif hasattr(base, "images"):
        names = [str(base.images[index][0]) for index in indices]
    else:
        names = [f"sample_{index:05d}" for index in indices]
    return names


class SatellitePairDataset(Dataset):
    """Paired HR/LR dataset with deterministic filename matching."""

    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int, training: bool) -> None:
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size
        self.training = training
        self.transform = _build_transform()
        self.pairs = self._collect_pairs()

    def _collect_pairs(self) -> list[tuple[str, Path, Path]]:
        hr_images = _collect_images(self.hr_dir)
        lr_images = _collect_images(self.lr_dir)
        if not hr_images or not lr_images:
            raise ValueError(f"No paired images found in {self.hr_dir} and {self.lr_dir}")
        hr_map = {_trace_name(self.hr_dir, path): path for path in hr_images}
        lr_map = {_trace_name(self.lr_dir, path): path for path in lr_images}
        common = sorted(set(hr_map) & set(lr_map))
        if not common:
            raise ValueError("No filename-aligned LR/HR pairs were found.")
        return [(name, hr_map[name], lr_map[name]) for name in common]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        name, hr_path, lr_path = self.pairs[index]
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        scale = hr_img.size[0] // lr_img.size[0]
        if self.training:
            hr_img, lr_img = self._random_crop_pair(hr_img, lr_img, scale)
        else:
            hr_img = transforms.CenterCrop(self.patch_size)(hr_img)
            lr_img = transforms.CenterCrop(self.patch_size // scale)(lr_img)
        if self.training and random.random() > 0.5:
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        return {
            "name": name,
            "scale": scale,
            "lr": self.transform(lr_img),
            "hr": self.transform(hr_img),
        }

    def _random_crop_pair(
        self,
        hr_img: Image.Image,
        lr_img: Image.Image,
        scale: int,
    ) -> tuple[Image.Image, Image.Image]:
        lr_crop = self.patch_size // scale
        lr_w, lr_h = lr_img.size
        if lr_w <= lr_crop or lr_h <= lr_crop:
            return hr_img, lr_img
        x = random.randint(0, lr_w - lr_crop)
        y = random.randint(0, lr_h - lr_crop)
        lr_box = (x, y, x + lr_crop, y + lr_crop)
        hr_box = (x * scale, y * scale, (x + lr_crop) * scale, (y + lr_crop) * scale)
        return hr_img.crop(hr_box), lr_img.crop(lr_box)


class SyntheticSatellitePairDataset(Dataset):
    """HR-only remote-sensing dataset with deterministic synthetic 4x LR generation."""

    def __init__(self, root_dir: str, patch_size: int, training: bool, scale_factor: int) -> None:
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.training = training
        self.scale_factor = scale_factor
        self.transform = _build_transform()
        self.images = self._collect_images()

    def _collect_images(self) -> list[tuple[str, Path]]:
        images = _collect_images(self.root_dir)
        if not images:
            raise ValueError(f"No HR images found under {self.root_dir}")
        return [(_trace_name(self.root_dir, path), path) for path in images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        name, hr_path = self.images[index]
        hr_img = Image.open(hr_path).convert("RGB")
        hr_crop = self._prepare_hr_crop(hr_img)
        lr_size = self.patch_size // self.scale_factor
        lr_img = hr_crop.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
        return {
            "name": name,
            "scale": self.scale_factor,
            "lr": self.transform(lr_img),
            "hr": self.transform(hr_crop),
        }

    def _prepare_hr_crop(self, hr_img: Image.Image) -> Image.Image:
        width, height = hr_img.size
        crop_size = min(self.patch_size, width, height)
        if self.training and width > crop_size and height > crop_size:
            x = random.randint(0, width - crop_size)
            y = random.randint(0, height - crop_size)
            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
        else:
            hr_crop = transforms.CenterCrop(crop_size)(hr_img)
        if crop_size != self.patch_size:
            hr_crop = hr_crop.resize((self.patch_size, self.patch_size), Image.Resampling.BICUBIC)
        if self.training and random.random() > 0.5:
            hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
        return hr_crop


def build_loaders(config: dict[str, Any], seed: int) -> DatasetBundle:
    """Build deterministic train/val loaders for paired or synthetic 4x datasets."""
    data = config["data"]
    spec = _resolve_spec(config)
    pairing_mode = spec["pairing_mode"]
    if pairing_mode == "paired":
        train_base = SatellitePairDataset(
            hr_dir=spec["hr_dir"],
            lr_dir=spec["lr_dir"],
            patch_size=data["patch_size"],
            training=True,
        )
        val_base = SatellitePairDataset(
            hr_dir=spec["hr_dir"],
            lr_dir=spec["lr_dir"],
            patch_size=data["patch_size"],
            training=False,
        )
    elif pairing_mode == "synthetic_4x":
        train_base = SyntheticSatellitePairDataset(
            root_dir=spec["root_dir"],
            patch_size=data["patch_size"],
            training=True,
            scale_factor=spec.get("scale_factor", data["fixed_scale"]),
        )
        val_base = SyntheticSatellitePairDataset(
            root_dir=spec["root_dir"],
            patch_size=data["patch_size"],
            training=False,
            scale_factor=spec.get("scale_factor", data["fixed_scale"]),
        )
    else:
        raise ValueError(f"Unsupported pairing mode '{pairing_mode}'")

    train_indices, val_indices = _split_indices(len(train_base), data["val_split"], seed)
    train_ds = _limited_subset(Subset(train_base, train_indices), data.get("train_limit"))
    val_ds = _limited_subset(Subset(val_base, val_indices), data.get("val_limit"))
    drop_last = len(train_ds) > data["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=data["num_workers"],
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data["num_workers"],
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_ds),
        val_size=len(val_ds),
        dataset_name=spec["name"],
        pairing_mode=pairing_mode,
        train_manifest=_names_from_subset(train_ds),
        val_manifest=_names_from_subset(val_ds),
    )


def build_split_manifest(bundle: DatasetBundle, config: dict[str, Any], version: str) -> dict[str, Any]:
    """Create a JSON-serializable split manifest."""
    return {
        "version": version,
        "dataset_family": config["dataset"]["family"],
        "dataset_name": bundle.dataset_name,
        "pairing_mode": bundle.pairing_mode,
        "seed": config["seed"],
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "train_manifest": bundle.train_manifest,
        "val_manifest": bundle.val_manifest,
    }
