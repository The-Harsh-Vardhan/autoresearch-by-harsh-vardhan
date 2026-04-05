"""Dataset loaders for tabular classification (Iris + Titanic)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from chakra.core.utils import REPO_ROOT


@dataclass
class TabularBundle:
    """Holds train / val dataloaders and metadata."""
    dataset_name: str
    train_loader: DataLoader
    val_loader: DataLoader
    feature_names: list[str]
    class_names: list[str]
    num_features: int
    num_classes: int
    train_size: int
    val_size: int


def _load_iris() -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """Load the Iris dataset from sklearn (bundled, no external file needed)."""
    from sklearn.datasets import load_iris
    ds = load_iris()
    features = ds.data.tolist()
    labels = ds.target.tolist()
    feature_names = list(ds.feature_names)
    class_names = list(ds.target_names)
    return features, labels, feature_names, class_names


def _load_titanic(data_path: Path) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """Load Titanic from a CSV file with basic feature engineering."""
    rows: list[dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
    class_names = ["died", "survived"]
    features: list[list[float]] = []
    labels: list[int] = []

    # Compute median age for imputation
    ages = [float(r["Age"]) for r in rows if r.get("Age", "").strip()]
    median_age = sorted(ages)[len(ages) // 2] if ages else 30.0

    for row in rows:
        try:
            pclass = float(row.get("Pclass", 2))
            sex = 1.0 if row.get("Sex", "").strip().lower() == "male" else 0.0
            age = float(row["Age"]) if row.get("Age", "").strip() else median_age
            sibsp = float(row.get("SibSp", 0))
            parch = float(row.get("Parch", 0))
            fare = float(row.get("Fare", 0)) if row.get("Fare", "").strip() else 0.0
            embarked = row.get("Embarked", "").strip()
            emb_c = 1.0 if embarked == "C" else 0.0
            emb_q = 1.0 if embarked == "Q" else 0.0
            emb_s = 1.0 if embarked == "S" else 0.0
            label = int(row.get("Survived", 0))
            features.append([pclass, sex, age, sibsp, parch, fare, emb_c, emb_q, emb_s])
            labels.append(label)
        except (ValueError, KeyError):
            continue  # skip malformed rows

    return features, labels, feature_names, class_names


def _normalize(features: list[list[float]]) -> list[list[float]]:
    """Simple z-score normalization."""
    if not features:
        return features
    n_feat = len(features[0])
    means = [sum(row[i] for row in features) / len(features) for i in range(n_feat)]
    stds = [max((sum((row[i] - means[i]) ** 2 for row in features) / len(features)) ** 0.5, 1e-8)
            for i in range(n_feat)]
    return [[(row[i] - means[i]) / stds[i] for i in range(n_feat)] for row in features]


def build_loaders(config: dict, seed: int = 42) -> TabularBundle:
    """Build train/val DataLoaders from the dataset specified in config."""
    dataset_name = config["data"]["dataset"]
    val_split = config["data"].get("val_split", 0.2)
    batch_size = config["data"].get("batch_size", 32)

    if dataset_name == "iris":
        features, labels, feature_names, class_names = _load_iris()
    elif dataset_name == "titanic":
        data_path = REPO_ROOT / config["data"]["data_file"]
        features, labels, feature_names, class_names = _load_titanic(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    features = _normalize(features)

    # Deterministic shuffle + split
    import random
    rng = random.Random(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = torch.tensor([features[i] for i in train_idx], dtype=torch.float32)
    y_train = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    X_val = torch.tensor([features[i] for i in val_idx], dtype=torch.float32)
    y_val = torch.tensor([labels[i] for i in val_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(class_names)

    return TabularBundle(
        dataset_name=dataset_name,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_names=feature_names,
        class_names=class_names,
        num_features=len(feature_names),
        num_classes=num_classes,
        train_size=len(train_idx),
        val_size=len(val_idx),
    )


def build_split_manifest(bundle: TabularBundle, config: dict, version: str) -> dict[str, Any]:
    """Build a JSON-serializable manifest of the dataset split."""
    return {
        "version": version,
        "dataset": bundle.dataset_name,
        "num_features": bundle.num_features,
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "feature_names": bundle.feature_names,
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "val_split": config["data"].get("val_split", 0.2),
        "batch_size": config["data"].get("batch_size", 32),
    }
