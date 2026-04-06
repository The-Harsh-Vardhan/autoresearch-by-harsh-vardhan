"""Evaluate a trained NLP language model."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from chakra.core.tracker import init_tracker
from chakra.core.utils import describe_run_dirs, get_device, load_config, load_dotenv, prepare_workspace_temp, set_seed, write_json

from .dataset import build_loaders, build_split_manifest
from .metrics import calculate_bpb, calculate_cross_entropy, calculate_perplexity
from .models import BigramBaseline, GPTNano


def build_model(config: dict, vocab_size: int, seq_len: int, device: torch.device, checkpoint: str | None) -> torch.nn.Module | None:
    """Instantiate and load a checkpointed model."""
    kind = config["model"]["kind"]
    if kind == "bigram":
        if checkpoint is None:
            return BigramBaseline(vocab_size=vocab_size).to(device)
        model = BigramBaseline(vocab_size=vocab_size).to(device)
    elif kind == "gpt_nano":
        if checkpoint is None:
            raise ValueError("A checkpoint is required for GPTNano evaluation.")
        model = GPTNano(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_embd=config["model"].get("n_embd", 64),
            n_head=config["model"].get("n_head", 4),
            n_layer=config["model"].get("n_layer", 4),
            dropout=0.0,  # No dropout during evaluation
        ).to(device)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate(config: dict, run_name: str, device: torch.device, checkpoint: str | None) -> dict[str, object]:
    """Evaluate a language model on the validation set."""
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

    model = build_model(config, bundle.vocab_size, bundle.seq_len, device, checkpoint)
    limit = config["evaluation"]["sample_limit"]
    total_loss = 0.0
    total_bpb = 0.0
    total_ppl = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(bundle.val_loader, desc="Evaluating", leave=False)):
            if batch_idx >= limit:
                break
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            logits = model(x)
            total_loss += calculate_cross_entropy(logits, y)
            total_bpb += calculate_bpb(logits, y)
            total_ppl += calculate_perplexity(logits, y)
            count += 1

    summary = {
        "run_name": run_name,
        "version": version,
        "model_kind": config["model"]["kind"],
        "dataset_name": bundle.dataset_name,
        "vocab_size": bundle.vocab_size,
        "seq_len": bundle.seq_len,
        "device": str(device),
        "checkpoint": checkpoint,
        "val_loss": total_loss / max(count, 1),
        "val_bpb": total_bpb / max(count, 1),
        "val_perplexity": total_ppl / max(count, 1),
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
    parser = argparse.ArgumentParser(description="Evaluate NLP language model")
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
    print(f"Evaluation complete: BPB={summary['val_bpb']:.4f}, Perplexity={summary['val_perplexity']:.2f}")


if __name__ == "__main__":
    main()
