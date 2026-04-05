"""Tabular Classification domain lifecycle hooks — implements DomainLifecycleHooks protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None

from chakra.core.interfaces import VersionPaths
from chakra.core.utils import REPO_ROOT, load_config

from .utils import version_slug, version_stem


class LifecycleHooks:
    """Tabular classification implementation of the DomainLifecycleHooks protocol."""

    # -- Naming helpers -------------------------------------------------------

    def version_stem(self, version: str) -> str:
        return version_stem(version)

    def version_slug(self, version: str) -> str:
        return version_slug(version)

    # -- Path resolution ------------------------------------------------------

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
                "control": resolved_root / "configs" / "tabular_cls" / f"{version}_control.yaml",
                "smoke": resolved_root / "configs" / "tabular_cls" / f"{version}_smoke.yaml",
                "train": resolved_root / "configs" / "tabular_cls" / f"{version}_train.yaml",
            },
            notebook_readme=resolved_root / "notebooks" / "versions" / "README.md",
            kernel_metadata=resolved_root / "notebooks" / "versions" / "kernel-metadata.json",
            next_ablation=resolved_root / "reports" / "generated" / f"{stem}.next_ablation.md",
            run_manifest=resolved_root / "artifacts" / "runs" / version / "run_manifest.json",
            kaggle_output_dir=resolved_root / "artifacts" / "kaggle_outputs" / version,
        )

    # -- Config generation ----------------------------------------------------

    def _base_config_template(self, dataset: str = "iris") -> dict[str, Any]:
        base = {
            "seed": 42,
            "project": {
                "name": "autoresearch-by-harsh-vardhan",
                "group": "tabular-v1",
                "tags": ["tabular", "classification"],
            },
            "runtime": {"version": "v1.0", "lineage": "scratch", "parent": None},
            "paths": {"artifact_root": "artifacts", "report_root": "reports"},
            "data": {
                "dataset": dataset,
                "val_split": 0.2,
                "batch_size": 32,
            },
            "tracking": {
                "enabled": True,
                "mode": "online",
                "project": "autoresearch-by-harsh-vardhan",
                "entity": None,
                "notes": "Tabular classification research lane",
            },
            "model": {"kind": "mlp", "hidden_dim": 64, "dropout": 0.2},
            "training": {
                "epochs": 30,
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
                "max_train_batches": None,
                "max_val_batches": None,
                "checkpoint_name": "v1.0_train_best.pt",
            },
            "evaluation": {"sample_limit": 100, "save_limit": 8},
        }
        if dataset == "titanic":
            base["data"]["data_file"] = "data/titanic.csv"
        return base

    def build_version_configs(
        self, version: str, parent: str | None, lineage: str,
    ) -> dict[str, dict[str, Any]]:
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
            tags = cfg["project"].get("tags", [])
            cfg["project"]["tags"] = list(dict.fromkeys([*tags, version, lineage]))

        # Control: logistic regression, 1 epoch, eval only
        control["project"]["group"] = f"{version}-control"
        control["project"]["tags"] = list(dict.fromkeys([*control["project"]["tags"], "control", "logistic"]))
        control["model"]["kind"] = "logistic"
        control["training"]["epochs"] = 1
        control["training"]["max_train_batches"] = 0
        control["training"]["checkpoint_name"] = f"{version}_control.pt"

        # Smoke: MLP, 3 epochs, limited batches
        smoke["project"]["group"] = f"{version}-smoke"
        smoke["project"]["tags"] = list(dict.fromkeys([*smoke["project"]["tags"], "smoke", "mlp"]))
        smoke["training"]["epochs"] = 3
        smoke["training"]["max_train_batches"] = 5
        smoke["training"]["max_val_batches"] = 3
        smoke["training"]["checkpoint_name"] = f"{version}_smoke_best.pt"

        # Train: MLP, full training
        train["project"]["group"] = f"{version}-train"
        train["project"]["tags"] = list(dict.fromkeys([*train["project"]["tags"], "train", "mlp"]))
        train["training"]["epochs"] = 30
        train["training"]["max_train_batches"] = None
        train["training"]["max_val_batches"] = None
        train["training"]["checkpoint_name"] = f"{version}_train_best.pt"

        return {"control": control, "smoke": smoke, "train": train}

    # -- Rendering ------------------------------------------------------------

    def render_notebook(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        setup_code = (
            f'import os\\nimport subprocess\\nimport sys\\nfrom pathlib import Path\\n\\n'
            f'def find_repo_root() -> Path:\\n'
            f'    for candidate in [Path.cwd(), Path("/kaggle/working")]:\\n'
            f'        if (candidate / "src" / "chakra").exists():\\n'
            f'            return candidate\\n'
            f'    return Path.cwd()\\n\\n'
            f'REPO_ROOT = find_repo_root()\\n'
            f'ENV = os.environ.copy()\\n'
            f'ENV["PYTHONPATH"] = str(REPO_ROOT / "src")\\n'
            f'VERSION = "{version}"\\n'
            f'print(f"Repo root: {{REPO_ROOT}}")\\n'
        )
        run_control = (
            f'subprocess.run([sys.executable, "-m", '
            f'"chakra.domains.tabular_cls.train_runner", '
            f'"--config", f"configs/tabular_cls/{version}_control.yaml", '
            f'"--run-name", f"{{VERSION}}-control"], '
            f'cwd=REPO_ROOT, env=ENV, check=True)\\n'
        )
        run_train = (
            f'subprocess.run([sys.executable, "-m", '
            f'"chakra.domains.tabular_cls.train_runner", '
            f'"--config", f"configs/tabular_cls/{version}_train.yaml", '
            f'"--run-name", f"{{VERSION}}-train"], '
            f'cwd=REPO_ROOT, env=ENV, check=True)\\n'
        )
        run_eval = (
            f'subprocess.run([sys.executable, "-m", '
            f'"chakra.domains.tabular_cls.evaluate_runner", '
            f'"--config", f"configs/tabular_cls/{version}_train.yaml", '
            f'"--run-name", f"{{VERSION}}-eval", '
            f'"--checkpoint", f"artifacts/{{VERSION}}-train/checkpoints/{version}_train_best.pt"], '
            f'cwd=REPO_ROOT, env=ENV, check=True)\\n'
        )
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [f"# {version} Tabular Classification\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": setup_code.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Control Baseline\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": run_control.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Full Training\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": run_train.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Evaluation\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": run_eval.splitlines(True)},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return json.dumps(notebook, indent=1)

    def render_doc(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        return (
            f"# {version} Tabular Classification\n\n"
            f"## Objective\n\n"
            f"Prove the tabular classification domain plugs into the AutoResearch lifecycle.\n\n"
            f"## Config Contract\n\n"
            f"- Control: `configs/tabular_cls/{version}_control.yaml`\n"
            f"- Smoke: `configs/tabular_cls/{version}_smoke.yaml`\n"
            f"- Train: `configs/tabular_cls/{version}_train.yaml`\n"
        )

    def render_review(self, version: str) -> str:
        return (
            f"# {version} Tabular Classification Review\n\n"
            f"## Status\n\n- State: pending run\n\n"
            f"## Findings\n\nPending execution.\n\n"
            f"## Promotion Decision\n\n- Decision:\n"
        )

    def render_notebook_readme(self) -> str:
        return (
            "# Versioned Tabular Notebooks\n\n"
            "- `v*.0_Tabular_CLS.ipynb` — tabular classification experiment notebooks.\n"
            "- Pair every notebook with doc and review files.\n"
        )

    def default_kernel_metadata(self) -> dict[str, Any]:
        return {
            "id": "INSERT_KAGGLE_USERNAME/tabular-cls-v1",
            "title": "Tabular Classification v1.0",
            "code_file": "v1.0_Tabular_CLS.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
        }

    # -- Review / analysis ---------------------------------------------------

    def build_findings(
        self, version: str, manifest: dict[str, Any], eval_summary: dict[str, Any] | None,
        registry: dict[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, float]]:
        findings: list[dict[str, str]] = []
        delta = {"accuracy_delta": 0.0, "f1_delta": 0.0}
        baselines = registry.get("control_baselines", {})
        baseline_entry = next(iter(baselines.values()), {}) if baselines else {}
        baseline_acc = baseline_entry.get("val_accuracy", 0.5)

        if not manifest.get("best_checkpoint"):
            findings.append({"severity": "high", "message": "Best checkpoint is missing from the synced output."})
        if not eval_summary:
            findings.append({"severity": "critical", "message": "No evaluation summary found."})
            return findings, delta

        val_acc = eval_summary.get("val_accuracy", 0.0)
        delta["accuracy_delta"] = round(val_acc - baseline_acc, 4)
        delta["f1_delta"] = round(eval_summary.get("val_f1", 0.0) - baseline_entry.get("val_f1", 0.0), 4)

        if delta["accuracy_delta"] <= 0:
            findings.append({"severity": "high", "message": f"Accuracy did not improve over baseline ({baseline_acc:.2%})."})
        else:
            findings.append({"severity": "low", "message": f"Accuracy improved by {delta['accuracy_delta']:.2%} over baseline."})

        if val_acc < 0.7:
            findings.append({"severity": "medium", "message": f"Absolute accuracy {val_acc:.2%} is below 70% threshold."})

        return findings, delta

    def ablation_suggestions(
        self, version: str, eval_summary: dict[str, Any] | None, delta: dict[str, float],
    ) -> list[str]:
        parts = version.split(".")
        if len(parts) == 2:
            major = f"v{int(parts[0][1:]) + 1}.0"
            minor = f"{version}.1"
        else:
            major, minor = f"{version}-next", f"{version}.1"
        if not eval_summary:
            return [f"{minor}: fix evaluation pipeline", f"{major}: try a different dataset"]
        suggestions = [
            f"{minor}: increase hidden_dim from 64 to 128",
            f"{major}: switch dataset from iris to titanic (more features, harder problem)",
        ]
        if delta.get("accuracy_delta", 0.0) <= 0:
            suggestions.insert(0, f"{minor}: increase epochs from 30 to 60")
        return suggestions[:3]

    def roast_lines(self) -> list[str]:
        return [
            "A logistic regression beats your MLP? That's not a model, that's a participation trophy.",
            "If your tabular model can't beat sklearn, rethink the architecture before adding layers.",
            "Iris has 150 samples. If you need 30 epochs, your learning rate is begging for help.",
        ]

    # -- Validation -----------------------------------------------------------

    def validate_version(self, version: str) -> list[str]:
        paths = self.resolve_version_paths(version)
        failures: list[str] = []
        if not paths.notebook.exists():
            failures.append(f"Missing notebook: {paths.notebook}")
        if not paths.doc.exists():
            failures.append(f"Missing doc: {paths.doc}")
        if not paths.review.exists():
            failures.append(f"Missing review: {paths.review}")
        for key, cfg_path in paths.configs.items():
            if not cfg_path.exists():
                failures.append(f"Missing {key} config: {cfg_path}")
        return failures
