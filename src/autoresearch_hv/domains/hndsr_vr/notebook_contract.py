"""Validation helpers for immutable versioned Kaggle notebooks."""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch_hv.core.tracker import init_tracker
from autoresearch_hv.core.utils import load_config, prepare_workspace_temp, repo_path


REQUIRED_NOTEBOOK_SECTIONS = (
    "## Runtime Compatibility Check",
    "## Post-Restart GPU Sanity Check",
    "## Experiment Registry",
    "## Paper Lineage and Hypothesis",
    "## Dataset and Config Contract",
    "## Weights & Biases Setup",
    "## Training Execution",
    "## Evaluation and Export",
    "## Results Dashboard",
    "## Troubleshooting and Known Failure Modes",
    "## Changelog",
    "## Next Step Gate",
)

REQUIRED_DOC_SECTIONS = (
    "## Objective",
    "## Kaggle Run Guide",
    "## Config Contract",
    "## Expected Artifacts",
    "## Handoff Back For Review",
)

REQUIRED_REVIEW_SECTIONS = (
    "## Status",
    "## Run Intake",
    "## Audit Checklist",
    "## Findings",
    "## Roast",
    "## Promotion Decision",
)


def _load_text(path: str | Path) -> str:
    return repo_path(path).read_text(encoding="utf-8")


def _load_notebook_text(path: str | Path) -> str:
    notebook = json.loads(repo_path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _missing_fragments(text: str, fragments: tuple[str, ...], label: str) -> list[str]:
    failures: list[str] = []
    for fragment in fragments:
        if fragment not in text:
            failures.append(f"{label} is missing fragment: {fragment}")
    return failures


def validate_versioned_notebook(
    notebook_path: str | Path,
    doc_path: str | Path,
    review_path: str | Path,
    full_config_path: str | Path,
    smoke_config_path: str | Path,
    control_config_path: str | Path,
    version: str,
) -> list[str]:
    """Validate the immutable notebook contract before Kaggle handoff."""
    failures: list[str] = []
    notebook_text = _load_notebook_text(notebook_path)
    doc_text = _load_text(doc_path)
    review_text = _load_text(review_path)

    required_notebook_sections = REQUIRED_NOTEBOOK_SECTIONS + (f"# {version} HNDSR",)
    required_notebook_commands = (
        f"python -m autoresearch_hv validate-version --version {version}",
        "python -m autoresearch_hv.hndsr_vr.train_runner --config",
        "python -m autoresearch_hv.hndsr_vr.evaluate_runner --config",
    )

    failures.extend(_missing_fragments(notebook_text, required_notebook_sections, "Notebook"))
    failures.extend(_missing_fragments(notebook_text, required_notebook_commands, "Notebook"))
    failures.extend(_missing_fragments(doc_text, REQUIRED_DOC_SECTIONS, "Doc"))
    failures.extend(_missing_fragments(review_text, REQUIRED_REVIEW_SECTIONS, "Review"))

    for config_path in (full_config_path, smoke_config_path, control_config_path):
        config = load_config(config_path)
        if config["dataset"]["name"] != "kaggle_4x":
            failures.append(f"{config_path} must target kaggle_4x for phase one.")
        if config["dataset"]["pairing_mode"] != "paired":
            failures.append(f"{config_path} must use paired LR/HR loading.")
        if config["dataset"]["scale_factor"] != 4:
            failures.append(f"{config_path} must keep a fixed 4x scale.")
        if config["tracking"].get("mode") not in {"offline", "online", "disabled"}:
            failures.append(f"{config_path} has unsupported tracking.mode.")
        if config["model"]["kind"] != "bicubic" and not config["training"].get("checkpoint_name"):
            failures.append(f"{config_path} must declare checkpoint_name for trainable runs.")

    full_config = load_config(full_config_path)
    prepare_workspace_temp(full_config["paths"]["artifact_root"])
    readiness_config = dict(full_config)
    readiness_tracking = dict(full_config.get("tracking", {}))
    readiness_tracking["enabled"] = False
    readiness_tracking["mode"] = "disabled"
    readiness_config["tracking"] = readiness_tracking
    tracker_dir = repo_path("artifacts/.tmp/notebook-readiness/tracker")
    tracker = init_tracker(readiness_config, f"{version}-readiness-check", tracker_dir)
    tracker.log_metrics({"readiness_contract": 1.0}, step=1)
    tracker.log_text("version", version)
    tracker.finish()

    try:
        from . import dataset as _dataset  # noqa: F401
        from autoresearch_hv.core import tracker as _tracker  # noqa: F401
        from autoresearch_hv.core import utils as _utils  # noqa: F401
    except Exception as exc:
        failures.append(f"Core HNDSR imports failed: {exc}")

    return failures
