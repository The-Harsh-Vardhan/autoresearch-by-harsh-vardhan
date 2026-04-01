"""Tracking helpers with W&B fallback and artifact support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir, flatten_config, prepare_workspace_temp, write_json


class NullTracker:
    """No-op tracker that still writes local run metadata."""

    backend = "null"

    def __init__(self, run_dir: str | Path, mode: str = "disabled", reason: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.reason = reason
        self.records: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.run_url: str | None = None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.records.append({"type": "metrics", "step": step, "values": metrics})

    def log_text(self, name: str, text: str) -> None:
        self.records.append({"type": "text", "name": name, "value": text})
        self.summary[name] = text

    def log_image(self, name: str, path: str | Path) -> None:
        self.records.append({"type": "image", "name": name, "path": str(path)})

    def log_file_artifact(self, name: str, path: str | Path, artifact_type: str, metadata: dict[str, Any] | None = None) -> None:
        self.records.append(
            {
                "type": "artifact_file",
                "name": name,
                "artifact_type": artifact_type,
                "path": str(path),
                "metadata": metadata or {},
            }
        )

    def log_dir_artifact(self, name: str, path: str | Path, artifact_type: str, metadata: dict[str, Any] | None = None) -> None:
        self.records.append(
            {
                "type": "artifact_dir",
                "name": name,
                "artifact_type": artifact_type,
                "path": str(path),
                "metadata": metadata or {},
            }
        )

    def finish(self) -> None:
        payload = {
            "backend": self.backend,
            "mode": self.mode,
            "reason": self.reason,
            "run_url": self.run_url,
            "summary": self.summary,
            "records": self.records,
        }
        write_json(self.run_dir / "tracker_records.json", payload)


class WandbTracker(NullTracker):
    """W&B-backed tracker with the same local fallback records."""

    backend = "wandb"

    def __init__(self, run_dir: str | Path, run: Any, mode: str) -> None:
        super().__init__(run_dir, mode=mode)
        self.run = run
        self._wandb = __import__("wandb")
        self.run_url = getattr(run, "url", None)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        if step is not None:
            self.run.log(dict(metrics), step=step)
        else:
            self.run.log(dict(metrics))

    def log_text(self, name: str, text: str) -> None:
        super().log_text(name, text)
        self.run.summary[name] = text

    def log_image(self, name: str, path: str | Path) -> None:
        super().log_image(name, path)
        self.run.log({name: self._wandb.Image(str(path))})

    def log_file_artifact(self, name: str, path: str | Path, artifact_type: str, metadata: dict[str, Any] | None = None) -> None:
        super().log_file_artifact(name, path, artifact_type, metadata)
        artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def log_dir_artifact(self, name: str, path: str | Path, artifact_type: str, metadata: dict[str, Any] | None = None) -> None:
        super().log_dir_artifact(name, path, artifact_type, metadata)
        artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_dir(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        super().finish()
        self.run.finish()


def init_tracker(config: dict[str, Any], run_name: str, run_dir: str | Path) -> NullTracker:
    """Initialize W&B if possible, otherwise return a local no-op tracker."""
    tracking = config["tracking"]
    mode = tracking.get("mode", "disabled")
    artifact_root = config.get("paths", {}).get("artifact_root", "artifacts")
    temp_root = prepare_workspace_temp(artifact_root)
    run_path = ensure_dir(run_dir)
    os.environ.setdefault("WANDB_DIR", str(run_path))
    os.environ.setdefault("WANDB_CACHE_DIR", str(ensure_dir(temp_root / "wandb-cache")))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(ensure_dir(temp_root / "wandb-config")))
    os.environ.setdefault("WANDB_DATA_DIR", str(ensure_dir(temp_root / "wandb-data")))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(ensure_dir(temp_root / "wandb-artifacts")))
    if mode == "disabled" or not tracking.get("enabled", True):
        return NullTracker(run_path, mode=mode)
    try:
        import wandb
    except Exception as exc:
        return NullTracker(run_path, mode=mode, reason=f"wandb import failed: {exc}")
    try:
        run = wandb.init(
            project=tracking["project"],
            entity=tracking.get("entity"),
            group=config["project"]["group"],
            tags=config["project"].get("tags", []),
            notes=tracking.get("notes"),
            name=run_name,
            config=flatten_config(config),
            mode=mode,
            reinit=True,
            dir=str(run_path),
        )
    except Exception as exc:
        return NullTracker(run_path, mode=mode, reason=f"wandb init failed: {exc}")
    return WandbTracker(run_path, run=run, mode=mode)
