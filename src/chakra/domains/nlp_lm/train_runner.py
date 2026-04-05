"""Train a language model for the NLP research domain."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from chakra.core.tracker import init_tracker
from chakra.core.utils import describe_run_dirs, get_device, load_config, load_dotenv, prepare_workspace_temp, set_seed, write_json

from .dataset import build_loaders, build_split_manifest
from .metrics import calculate_bpb, calculate_cross_entropy, calculate_perplexity
from .models import BigramBaseline, GPTNano


def build_model(config: dict, vocab_size: int, seq_len: int, device: torch.device) -> torch.nn.Module:
    """Instantiate the selected model."""
    kind = config["model"]["kind"]
    if kind == "gpt_nano":
        model = GPTNano(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_embd=config["model"].get("n_embd", 64),
            n_head=config["model"].get("n_head", 4),
            n_layer=config["model"].get("n_layer", 4),
            dropout=config["model"].get("dropout", 0.1),
        )
    elif kind == "bigram":
        model = BigramBaseline(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return model.to(device)


def validate(model: torch.nn.Module, val_loader, device: torch.device, max_batches: int | None) -> dict[str, float]:
    """Run a bounded validation pass."""
    model.eval()
    total_loss = 0.0
    total_bpb = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            logits = model(x)
            total_loss += calculate_cross_entropy(logits, y)
            total_bpb += calculate_bpb(logits, y)
            count += 1
    return {
        "val_loss": total_loss / max(count, 1),
        "val_bpb": total_bpb / max(count, 1),
        "val_perplexity": float(2 ** (total_bpb / max(count, 1))),
    }


def train(config: dict, run_name: str, device: torch.device) -> dict[str, object]:
    """Train the selected model and persist the best checkpoint."""
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

    model = build_model(config, bundle.vocab_size, bundle.seq_len, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    best_val_loss = float("inf")
    best_checkpoint = dirs["checkpoints"] / config["training"]["checkpoint_name"]
    history: list[dict[str, float]] = []
    max_train_batches = config["training"].get("max_train_batches")
    max_val_batches = config["training"].get("max_val_batches")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(bundle.train_loader, desc=f"LM epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(progress):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
            loss.backward()
            clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            train_losses.append(float(loss.item()))
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        val_metrics = validate(model, bundle.val_loader, device, max_val_batches)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            **val_metrics,
        }
        history.append(epoch_metrics)
        tracker.log_metrics(epoch_metrics, step=epoch + 1)
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            torch.save({"model": model.state_dict(), "config": config, "vocab_size": bundle.vocab_size}, best_checkpoint)

    tracker.log_file_artifact(f"{run_name}-best-checkpoint", best_checkpoint, "checkpoint")
    summary = {
        "run_name": run_name,
        "version": version,
        "model_kind": config["model"]["kind"],
        "dataset_name": bundle.dataset_name,
        "vocab_size": bundle.vocab_size,
        "seq_len": bundle.seq_len,
        "num_params": num_params,
        "device": str(device),
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "best_checkpoint": str(best_checkpoint),
        "best_val_loss": best_val_loss,
        "tracker_backend": tracker.backend,
        "tracker_url": tracker.run_url,
        "history": history,
    }
    summary_path = write_json(dirs["metrics"] / "train_summary.json", summary)
    tracker.log_text("best_checkpoint", str(best_checkpoint))
    tracker.log_file_artifact(f"{run_name}-train-summary", summary_path, "metrics_summary")
    tracker.finish()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NLP language model")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--device", default=None, help="Optional torch device override")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    prepare_workspace_temp(config["paths"]["artifact_root"])
    set_seed(config["seed"])
    run_name = args.run_name or f"{config['project']['group']}-{time.strftime('%Y%m%d-%H%M%S')}"
    device = get_device(args.device)
    summary = train(config, run_name, device)
    print(f"Saved best checkpoint to {summary['best_checkpoint']}")


if __name__ == "__main__":
    main()
