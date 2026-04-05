"""NLP domain lifecycle hooks — implements DomainLifecycleHooks protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None

from autoresearch_hv.core.interfaces import VersionPaths
from autoresearch_hv.core.utils import REPO_ROOT, load_config

from .utils import version_slug, version_stem


class LifecycleHooks:
    """NLP implementation of the DomainLifecycleHooks protocol."""

    def version_stem(self, version: str) -> str:
        return version_stem(version)

    def version_slug(self, version: str) -> str:
        return version_slug(version)

    def resolve_version_paths(self, version: str, root: Path | None = None) -> VersionPaths:
        resolved_root = root or REPO_ROOT
        stem = self.version_stem(version)
        return VersionPaths(
            root=resolved_root,
            version=version,
            stem=stem,
            notebook=resolved_root / "notebooks" / "versions" / f"{stem}.ipynb",
            doc=resolved_root / "docs" / "notebooks" / f"{stem}.md",
            review=resolved_root / "reports" / "reviews" / f"{stem}.review.md",
            review_json=resolved_root / "reports" / "reviews" / f"{stem}.review.json",
            configs={
                "control": resolved_root / "configs" / "nlp_lm" / f"{version}_control.yaml",
                "smoke": resolved_root / "configs" / "nlp_lm" / f"{version}_smoke.yaml",
                "train": resolved_root / "configs" / "nlp_lm" / f"{version}_train.yaml",
            },
            notebook_readme=resolved_root / "notebooks" / "versions" / "README.md",
            kernel_metadata=resolved_root / "notebooks" / "versions" / "kernel-metadata.json",
            next_ablation=resolved_root / "reports" / "generated" / f"{stem}.next_ablation.md",
            run_manifest=resolved_root / "artifacts" / "runs" / version / "run_manifest.json",
            kaggle_output_dir=resolved_root / "artifacts" / "kaggle_outputs" / version,
        )

    def _base_config_template(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "project": {"name": "autoresearch-by-harsh-vardhan", "group": "nlp-lm", "tags": ["phase1", "nlp"]},
            "runtime": {"version": "v1.0", "lineage": "scratch", "parent": None},
            "paths": {
                "artifact_root": "artifacts",
                "report_root": "reports",
                "datasets": {
                    "tiny_shakespeare": {
                        "text_file": "data/tiny_shakespeare.txt",
                    }
                },
            },
            "dataset": {"name": "tiny_shakespeare"},
            "data": {"seq_len": 128, "batch_size": 32, "val_split": 0.1, "train_limit": None, "val_limit": None},
            "tracking": {"enabled": True, "mode": "online", "project": "autoresearch-by-harsh-vardhan", "entity": None, "notes": "NLP LM research lane"},
            "model": {"kind": "gpt_nano", "n_embd": 64, "n_head": 4, "n_layer": 4, "dropout": 0.1},
            "training": {"epochs": 5, "lr": 3.0e-4, "weight_decay": 1.0e-2, "grad_clip": 1.0, "max_train_batches": None, "max_val_batches": None, "checkpoint_name": "v1.0_train_best.pt"},
            "evaluation": {"sample_limit": 50},
        }

    def build_version_configs(self, version: str, parent: str | None, lineage: str) -> dict[str, dict[str, Any]]:
        if parent:
            parent_paths = self.resolve_version_paths(parent)
            control = load_config(parent_paths.configs["control"])
            smoke = load_config(parent_paths.configs["smoke"])
            train = load_config(parent_paths.configs["train"])
        else:
            train = self._base_config_template()
            smoke = copy.deepcopy(train)
            control = copy.deepcopy(train)

        for cfg in (control, smoke, train):
            cfg["runtime"] = {"version": version, "lineage": lineage, "parent": parent}
            tags = [tag for tag in cfg["project"].get("tags", []) if not tag.startswith("v")]
            cfg["project"]["tags"] = list(dict.fromkeys([*tags, version, lineage]))

        # Control: bigram baseline
        control["project"]["group"] = f"{version}-control"
        control["project"]["tags"] = list(dict.fromkeys([*control["project"]["tags"], "control", "bigram"]))
        control["model"]["kind"] = "bigram"
        control["training"]["epochs"] = 1
        control["training"]["max_train_batches"] = 0
        control["training"]["checkpoint_name"] = f"{version}_control.pt"

        # Smoke: tiny fast run
        smoke["project"]["group"] = f"{version}-smoke"
        smoke["project"]["tags"] = list(dict.fromkeys([*smoke["project"]["tags"], "smoke", "gpt_nano"]))
        smoke["training"]["epochs"] = 1
        smoke["training"]["max_train_batches"] = 10
        smoke["training"]["max_val_batches"] = 5
        smoke["data"]["batch_size"] = 8
        smoke["training"]["checkpoint_name"] = f"{version}_smoke_best.pt"

        # Train: full run
        train["project"]["group"] = f"{version}-train"
        train["project"]["tags"] = list(dict.fromkeys([*train["project"]["tags"], "train", "gpt_nano"]))
        train["training"]["checkpoint_name"] = f"{version}_train_best.pt"

        return {"control": control, "smoke": smoke, "train": train}

    def render_notebook(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        setup_lines = [
            "import os\n",
            "import sys\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            "def find_repo_root() -> Path:\n",
            '    candidates = [Path.cwd(), Path("/kaggle/working")]\n',
            "    for c in candidates:\n",
            '        if (c / "src" / "autoresearch_hv").exists():\n',
            "            return c\n",
            "    return Path.cwd()\n",
            "\n",
            "REPO_ROOT = find_repo_root()\n",
            "ENV = os.environ.copy()\n",
            'ENV["PYTHONPATH"] = str(REPO_ROOT / "src")\n',
            f'VERSION = "{version}"\n',
            'print(f"Repo root: {REPO_ROOT}")\n',
            "assert (REPO_ROOT / 'src' / 'autoresearch_hv').exists()\n",
        ]
        validate_lines = [
            "subprocess.run(\n",
            '    [sys.executable, "-m", "autoresearch_hv",\n',
            f'     "--domain", "nlp_lm", "validate-version", "--version", "{version}"],\n',
            "    cwd=REPO_ROOT, env=ENV, check=True,\n",
            ")\n",
        ]
        control_lines = [
            "subprocess.run(\n",
            f'    [sys.executable, "-m", "autoresearch_hv.domains.nlp_lm.train_runner",\n',
            f'     "--config", "configs/nlp_lm/{version}_control.yaml",\n',
            f'     "--run-name", "{version}-control"],\n',
            "    cwd=REPO_ROOT, env=ENV, check=True,\n",
            ")\n",
        ]
        smoke_lines = [
            "subprocess.run(\n",
            f'    [sys.executable, "-m", "autoresearch_hv.domains.nlp_lm.train_runner",\n',
            f'     "--config", "configs/nlp_lm/{version}_smoke.yaml",\n',
            f'     "--run-name", "{version}-smoke"],\n',
            "    cwd=REPO_ROOT, env=ENV, check=True,\n",
            ")\n",
        ]
        train_lines = [
            "subprocess.run(\n",
            f'    [sys.executable, "-m", "autoresearch_hv.domains.nlp_lm.train_runner",\n',
            f'     "--config", "configs/nlp_lm/{version}_train.yaml",\n',
            f'     "--run-name", "{version}-train"],\n',
            "    cwd=REPO_ROOT, env=ENV, check=True,\n",
            ")\n",
        ]
        eval_lines = [
            "subprocess.run(\n",
            f'    [sys.executable, "-m", "autoresearch_hv.domains.nlp_lm.evaluate_runner",\n',
            f'     "--config", "configs/nlp_lm/{version}_train.yaml",\n',
            f'     "--run-name", "{version}-eval",\n',
            f'     "--checkpoint", "artifacts/runs/{version}/checkpoints/{version}_train_best.pt"],\n',
            "    cwd=REPO_ROOT, env=ENV, check=True,\n",
            ")\n",
        ]

        def _code_cell(source_lines: list[str]) -> dict:
            return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines}

        def _md_cell(lines: list[str]) -> dict:
            return {"cell_type": "markdown", "metadata": {}, "source": lines}

        lineage_note = f"Fork of `{parent}`" if parent else "Scratch run"
        notebook = {
            "cells": [
                _md_cell([f"# {version} NLP LM\n", "\n", f"**{lineage_note}**\n"]),
                _md_cell(["## Runtime Compatibility Check\n"]),
                _code_cell(setup_lines),
                _md_cell(["## Weights & Biases Setup\n",
                           "\n",
                           "Ensure W&B is configured before running training cells.\n"]),
                _md_cell([f"## Experiment Registry\n",
                           "\n",
                           f"- Version: `{version}`\n",
                           f"- Domain: `nlp_lm`\n",
                           f"- Lineage: {lineage_note}\n",
                           f"- Primary Metric: BPB ↓\n"]),
                _md_cell(["## Version Contract Validation\n"]),
                _code_cell(validate_lines),
                _md_cell(["## Control Baseline (Bigram)\n"]),
                _code_cell(control_lines),
                _md_cell(["## Smoke Test\n"]),
                _code_cell(smoke_lines),
                _md_cell(["## Full Training Run\n"]),
                _code_cell(train_lines),
                _md_cell(["## Evaluation\n"]),
                _code_cell(eval_lines),
                _md_cell(["## Results Dashboard\n",
                           "\n",
                           "Review outputs in W&B and `artifacts/runs/` directory.\n"]),
                _md_cell(["## Changelog\n", "\n", "- Initial scaffold.\n"]),
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return json.dumps(notebook, indent=1)

    def render_doc(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        lineage = f"Fork of `{parent}`" if parent else "Scratch run"
        return "\n".join([
            f"# {version} NLP LM",
            "",
            "## Objective",
            "",
            f"`{version}` trains a character-level GPT-nano language model on Tiny Shakespeare.",
            f"Lineage: {lineage}.",
            "",
            "## Config Contract",
            "",
            f"- Train config: `configs/nlp_lm/{version}_train.yaml`",
            f"- Smoke config: `configs/nlp_lm/{version}_smoke.yaml`",
            f"- Control config: `configs/nlp_lm/{version}_control.yaml`",
            "",
            "## Expected Artifacts",
            "",
            f"- Best checkpoint: `artifacts/runs/{version}/checkpoints/{version}_train_best.pt`",
            f"- Train summary: `artifacts/runs/{version}/metrics/train_summary.json`",
            f"- Eval summary: `artifacts/runs/{version}/metrics/eval_summary.json`",
            "",
            "## Metrics",
            "",
            "- Primary: **BPB** (bits per byte) — lower is better",
            "- Secondary: cross-entropy loss, perplexity",
            "",
        ])

    def render_review(self, version: str) -> str:
        return "\n".join([
            f"# {version} NLP LM Review",
            "",
            "## Status",
            "",
            "- State: pending run",
            "- Domain: NLP Language Modelling",
            "",
            "## Run Intake",
            "",
            "- Best checkpoint path:",
            "- Eval summary path:",
            "- W&B URL:",
            "",
            "## Audit Checklist",
            "",
            "- [ ] Model code is repo-owned",
            "- [ ] Metrics are traceable to a versioned config",
            "- [ ] BPB beats bigram baseline",
            "- [ ] No data leakage between train/val splits",
            "",
            "## Findings",
            "",
            "Pending.",
            "",
            "## Roast",
            "",
            "Pending.",
            "",
            "## Promotion Decision",
            "",
            "- Decision:",
            "- Rationale:",
            "",
        ])

    def render_notebook_readme(self) -> str:
        return "\n".join([
            "# Versioned Notebooks",
            "",
            "Each NLP language modelling version gets its own notebook:",
            "",
            "- `vN.M_NLP_LM.ipynb` — Kaggle/Colab-ready notebook",
            "- Pair every notebook with `docs/notebooks/vN.M_NLP_LM.md` and `reports/reviews/vN.M_NLP_LM.review.md`",
            "",
        ])

    def default_kernel_metadata(self) -> dict[str, Any]:
        return {
            "id": "INSERT_KAGGLE_USERNAME/nlp-lm",
            "title": "NLP LM Baseline",
            "code_file": "v1.0_NLP_LM.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }

    def build_findings(
        self, version: str, manifest: dict[str, Any], eval_summary: dict[str, Any] | None, registry: dict[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, float]]:
        baseline = registry.get("control_baselines", {}).get("bigram_baseline", {})
        findings: list[dict[str, str]] = []
        delta = {"bpb_delta": 0.0}
        if not manifest.get("wandb_url"):
            findings.append({"severity": "medium", "message": "W&B URL is missing."})
        if not eval_summary:
            findings.append({"severity": "critical", "message": "No evaluation summary found."})
            return findings, delta
        baseline_bpb = baseline.get("val_bpb", float("inf"))
        model_bpb = eval_summary.get("val_bpb", float("inf"))
        delta["bpb_delta"] = round(baseline_bpb - model_bpb, 4)  # positive = model is better (lower BPB)
        if delta["bpb_delta"] <= 0:
            findings.append({"severity": "high", "message": f"BPB is worse than bigram baseline by {abs(delta['bpb_delta']):.4f}."})
        else:
            findings.append({"severity": "low", "message": f"BPB beats bigram baseline by {delta['bpb_delta']:.4f}."})
        return findings, delta

    def ablation_suggestions(
        self, version: str, eval_summary: dict[str, Any] | None, delta: dict[str, float],
    ) -> list[str]:
        parts = version.split(".")
        try:
            minor = f"v{parts[0][1:]}.{int(parts[1]) + 1}"
        except (IndexError, ValueError):
            minor = f"{version}.1"
        if not eval_summary:
            return [f"{minor}: fix missing evaluation before any model change"]
        return [
            f"{minor}: increase n_layer from 4 to 6",
            f"{minor}: increase n_embd from 64 to 128",
            f"{minor}: try learning rate 1e-3",
        ][:3]

    def roast_lines(self) -> list[str]:
        return [
            "A character-level LM that cannot beat a bigram is not a model, it is a random number generator.",
            "Training for 5 epochs on tiny data is not research, it is a unit test.",
        ]

    def validate_version(self, version: str) -> list[str]:
        paths = self.resolve_version_paths(version)
        failures: list[str] = []
        for name, config_path in paths.configs.items():
            if not config_path.exists():
                failures.append(f"Missing {name} config: {config_path}")
        if not paths.notebook.exists():
            failures.append(f"Missing notebook: {paths.notebook}")
        if not paths.doc.exists():
            failures.append(f"Missing doc: {paths.doc}")
        if not paths.review.exists():
            failures.append(f"Missing review: {paths.review}")
        return failures
