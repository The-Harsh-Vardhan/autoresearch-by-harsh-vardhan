"""Dataset helpers for the NLP language-modelling domain."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class DatasetBundle:
    """Container for NLP loaders and dataset metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int
    dataset_name: str
    vocab_size: int
    seq_len: int


class CharTextDataset(Dataset):
    """Character-level text dataset for small-scale LM experiments.

    Reads a plain-text file, builds a character vocabulary, and yields
    fixed-length token sequences for next-token prediction.
    """

    def __init__(self, text: str, seq_len: int, char_to_idx: dict[str, int] | None = None) -> None:
        self.seq_len = seq_len
        if char_to_idx is None:
            chars = sorted(set(text))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        else:
            self.char_to_idx = char_to_idx
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.data = torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = self.data[index : index + self.seq_len]
        y = self.data[index + 1 : index + self.seq_len + 1]
        return {"input_ids": x, "target_ids": y}


def _split_text(text: str, val_split: float, seed: int) -> tuple[str, str]:
    """Split text at a deterministic boundary."""
    split_point = max(1, int(len(text) * (1.0 - val_split)))
    return text[:split_point], text[split_point:]


def _load_text_file(path: str | Path) -> str:
    """Read a plain-text file."""
    return Path(path).read_text(encoding="utf-8")


def build_loaders(config: dict[str, Any], seed: int) -> DatasetBundle:
    """Build deterministic train/val loaders for a character-level text dataset."""
    data_cfg = config["data"]
    dataset_cfg = config["dataset"]
    text_path = config["paths"]["datasets"][dataset_cfg["name"]]["text_file"]
    text = _load_text_file(text_path)

    train_text, val_text = _split_text(text, data_cfg["val_split"], seed)

    train_ds = CharTextDataset(train_text, seq_len=data_cfg["seq_len"])
    val_ds = CharTextDataset(val_text, seq_len=data_cfg["seq_len"], char_to_idx=train_ds.char_to_idx)

    train_limit = data_cfg.get("train_limit")
    val_limit = data_cfg.get("val_limit")
    if train_limit and train_limit < len(train_ds):
        train_ds = Subset(train_ds, list(range(train_limit)))
    if val_limit and val_limit < len(val_ds):
        val_ds = Subset(val_ds, list(range(val_limit)))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=g,
        drop_last=len(train_ds) > data_cfg["batch_size"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    vocab_size = train_ds.vocab_size if hasattr(train_ds, "vocab_size") else train_ds.dataset.vocab_size
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_ds),
        val_size=len(val_ds),
        dataset_name=dataset_cfg["name"],
        vocab_size=vocab_size,
        seq_len=data_cfg["seq_len"],
    )


def build_split_manifest(bundle: DatasetBundle, config: dict[str, Any], version: str) -> dict[str, Any]:
    """Create a JSON-serializable split manifest."""
    return {
        "version": version,
        "dataset_name": bundle.dataset_name,
        "vocab_size": bundle.vocab_size,
        "seq_len": bundle.seq_len,
        "seed": config["seed"],
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
    }
