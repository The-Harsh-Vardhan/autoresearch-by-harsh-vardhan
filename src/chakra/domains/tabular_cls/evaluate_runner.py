"""Evaluate a trained tabular classification model."""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from chakra.core.tracker import init_tracker
from chakra.core.utils import (
    describe_run_dirs,
    get_device,
    load_config,
    load_dotenv,
    prepare_workspace_temp,
    set_seed,
    write_json,
)

from .dataset import build_loaders, build_split_manifest
from .metrics import calculate_accuracy, calculate_cross_entropy, calculate_f1
from .models import LogisticBaseline, SmallMLP


def build_model(config: dict, num_features: int, num_classes: int,
                device: torch.device, checkpoint: str | None) -> torch.nn.Module:
    """Instantiate and optionally load a checkpointed model."""
    kind = config["model"]["kind"]
    if kind == "logistic":
        model = LogisticBaseline(num_features, num_classes)
    elif kind == "mlp":
        model = SmallMLP(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=config["model"].get("hidden_dim", 64),
            dropout=0.0,  # no dropout at eval time
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model = model.to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate(config: dict, run_name: str, device: torch.device, checkpoint: str | None) -> dict[str, object]:
    """Evaluate a trained model on the validation set."""
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

    model = build_model(config, bundle.num_features, bundle.num_classes, device, checkpoint)
    sample_limit = config["evaluation"]["sample_limit"]
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(bundle.val_loader, desc="Evaluating", leave=False)):
            if batch_idx >= sample_limit:
                break
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += calculate_cross_entropy(logits, y)
            total_acc += calculate_accuracy(logits, y)
            total_f1 += calculate_f1(logits, y, bundle.num_classes)
            count += 1

    summary = {
        "run_name": run_name,
        "version": version,
        "model_kind": config["model"]["kind"],
        "dataset_name": bundle.dataset_name,
        "num_features": bundle.num_features,
        "num_classes": bundle.num_classes,
        "device": str(device),
        "checkpoint": checkpoint,
        "val_loss": total_loss / max(count, 1),
        "val_accuracy": total_acc / max(count, 1),
        "val_f1": total_f1 / max(count, 1),
        "num_samples": count,
        "tracker_backend": tracker.backend,
        "tracker_url": tracker.run_url,
    }
    tracker.log_metrics(summary)
    summary_path = write_json(dirs["metrics"] / "eval_summary.json", summary)
    tracker.log_file_artifact(f"{run_name}-eval-summary", summary_path, "metrics_summary")
    tracker.finish()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tabular classification model")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--device", default=None, help="Optional torch device override")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    prepare_workspace_temp(config["paths"]["artifact_root"])
    set_seed(config["seed"])
    run_name = args.run_name or f"{config['project']['group']}-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    device = get_device(args.device)
    summary = evaluate(config, run_name, device, args.checkpoint)
    print(f"Evaluation complete: Accuracy={summary['val_accuracy']:.2%}, F1={summary['val_f1']:.4f}")
    if summary.get("tracker_url"):
        print(f"W&B run: {summary['tracker_url']}")


if __name__ == "__main__":
    main()
