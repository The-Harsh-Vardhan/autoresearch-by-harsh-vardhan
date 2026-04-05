"""HNDSR domain lifecycle hooks — implements DomainLifecycleHooks protocol."""

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

from .notebook_contract import validate_versioned_notebook
from .utils import version_slug, version_stem


class LifecycleHooks:
    """HNDSR implementation of the DomainLifecycleHooks protocol."""

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
                "control": resolved_root / "configs" / "hndsr_vr" / f"{version}_control.yaml",
                "smoke": resolved_root / "configs" / "hndsr_vr" / f"{version}_smoke.yaml",
                "train": resolved_root / "configs" / "hndsr_vr" / f"{version}_train.yaml",
            },
            notebook_readme=resolved_root / "notebooks" / "versions" / "README.md",
            kernel_metadata=resolved_root / "notebooks" / "versions" / "kernel-metadata.json",
            next_ablation=resolved_root / "reports" / "generated" / f"{stem}.next_ablation.md",
            run_manifest=resolved_root / "artifacts" / "runs" / version / "run_manifest.json",
            kaggle_output_dir=resolved_root / "artifacts" / "kaggle_outputs" / version,
        )

    # -- Config generation ----------------------------------------------------

    def _base_config_template(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "project": {"name": "autoresearch-by-harsh-vardhan", "group": "hndsr-vR", "tags": ["phase1", "scratch"]},
            "runtime": {"version": "vR.1", "lineage": "scratch", "parent": None},
            "paths": {
                "artifact_root": "artifacts",
                "report_root": "reports",
                "datasets": {
                    "kaggle_4x": {
                        "hr_dir": "C:\\Users\\harsh\\.cache\\kagglehub\\datasets\\cristobaltudela\\4x-satellite-image-super-resolution\\versions\\1\\HR_0.5m\\HR_0.5m",
                        "lr_dir": "C:\\Users\\harsh\\.cache\\kagglehub\\datasets\\cristobaltudela\\4x-satellite-image-super-resolution\\versions\\1\\LR_2m\\LR_2m",
                    }
                },
            },
            "dataset": {"family": "kaggle", "name": "kaggle_4x", "pairing_mode": "paired", "scale_factor": 4},
            "data": {"patch_size": 64, "batch_size": 4, "num_workers": 0, "val_split": 0.1, "fixed_scale": 4, "train_limit": None, "val_limit": 8},
            "tracking": {"enabled": True, "mode": "online", "project": "autoresearch-by-harsh-vardhan", "entity": None, "notes": "HNDSR scratch research lane"},
            "model": {"kind": "sr3", "model_channels": 32},
            "training": {"epochs": 6, "lr": 1.0e-4, "weight_decay": 1.0e-4, "grad_clip": 1.0, "max_train_batches": None, "max_val_batches": 8, "checkpoint_name": "vR.1_train_best.pt"},
            "diffusion": {"num_timesteps": 1000, "beta_start": 1.0e-4, "beta_end": 0.02, "inference_steps": 10},
            "evaluation": {"sample_limit": 8, "save_limit": 8, "compute_lpips": True, "grid_name": "vR.1_grid.png"},
        }

    def build_version_configs(self, version: str, parent: str | None, lineage: str) -> dict[str, dict[str, Any]]:
        if lineage != "scratch":
            raise ValueError("Phase one ships only the scratch HNDSR lineage.")
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
            cfg["project"]["name"] = "autoresearch-by-harsh-vardhan"
            cfg["tracking"]["project"] = "autoresearch-by-harsh-vardhan"
            tags = [tag for tag in cfg["project"].get("tags", []) if not str(tag).startswith("vR")]
            cfg["project"]["tags"] = list(dict.fromkeys([*tags, version, lineage, "kaggle"]))
        control["project"]["group"] = f"{version}-control"
        control["project"]["tags"] = list(dict.fromkeys([*control["project"]["tags"], "control", "bicubic"]))
        control["model"]["kind"] = "bicubic"
        control["training"]["epochs"] = 1
        control["training"]["max_train_batches"] = 0
        control["training"]["checkpoint_name"] = f"{version}_control.pt"
        control["evaluation"]["grid_name"] = f"{version}_control_grid.png"

        smoke["project"]["group"] = f"{version}-smoke"
        smoke["project"]["tags"] = list(dict.fromkeys([*smoke["project"]["tags"], "smoke", "sr3"]))
        smoke["training"]["epochs"] = 1
        smoke["training"]["max_train_batches"] = 4
        smoke["training"]["max_val_batches"] = 2
        smoke["data"]["batch_size"] = 2
        smoke["data"]["val_limit"] = 4
        smoke["training"]["checkpoint_name"] = f"{version}_smoke_best.pt"
        smoke["evaluation"]["grid_name"] = f"{version}_smoke_grid.png"

        train["project"]["group"] = f"{version}-train"
        train["project"]["tags"] = list(dict.fromkeys([*train["project"]["tags"], "train", "sr3"]))
        train["training"]["epochs"] = 6
        train["training"]["max_train_batches"] = None
        train["training"]["max_val_batches"] = 8
        train["data"]["batch_size"] = 4
        train["data"]["val_limit"] = 16
        train["training"]["checkpoint_name"] = f"{version}_train_best.pt"
        train["evaluation"]["grid_name"] = f"{version}_train_grid.png"
        return {"control": control, "smoke": smoke, "train": train}

    # -- Rendering ------------------------------------------------------------

    def render_notebook(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        setup_code = f'''import os\nimport platform\nimport subprocess\nimport sys\nfrom pathlib import Path\n\n\ndef find_repo_root() -> Path:\n    candidates = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]\n    for candidate in candidates:\n        if (candidate / "src" / "autoresearch_hv").exists():\n            return candidate\n    return Path.cwd()\n\n\nREPO_ROOT = find_repo_root()\nENV = os.environ.copy()\nENV["PYTHONPATH"] = str(REPO_ROOT / "src")\nVERSION = "{version}"\nPARENT = {repr(parent)}\nCONTROL_CONFIG = "configs/hndsr_vr/{version}_control.yaml"\nSMOKE_CONFIG = "configs/hndsr_vr/{version}_smoke.yaml"\nTRAIN_CONFIG = "configs/hndsr_vr/{version}_train.yaml"\nprint(f"Repo root: {{REPO_ROOT}}")\nprint(f"Python: {{sys.executable}}")\nprint(f"Platform: {{platform.platform()}}")\nprint(f"Kaggle runtime type: {{os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'local-or-unknown')}}")\nassert (REPO_ROOT / 'src' / 'autoresearch_hv').exists(), 'Expected src/autoresearch_hv under repo root.'\n'''
        run_control = f'''subprocess.run([sys.executable, "-m", "autoresearch_hv.hndsr_vr.evaluate_runner", "--config", CONTROL_CONFIG, "--run-name", f"{{VERSION}}-control"], cwd=REPO_ROOT, env=ENV, check=True)'''
        run_smoke = f'''subprocess.run([sys.executable, "-m", "autoresearch_hv.hndsr_vr.train_runner", "--config", SMOKE_CONFIG, "--run-name", f"{{VERSION}}-smoke"], cwd=REPO_ROOT, env=ENV, check=True)\nsubprocess.run([sys.executable, "-m", "autoresearch_hv.hndsr_vr.evaluate_runner", "--config", SMOKE_CONFIG, "--run-name", f"{{VERSION}}-smoke-eval", "--checkpoint", f"artifacts/{{VERSION}}-smoke/checkpoints/{version}_smoke_best.pt"], cwd=REPO_ROOT, env=ENV, check=True)'''
        run_train = f'''subprocess.run([sys.executable, "-m", "autoresearch_hv", "validate-version", "--version", VERSION], cwd=REPO_ROOT, env=ENV, check=True)\nsubprocess.run([sys.executable, "-m", "autoresearch_hv.hndsr_vr.train_runner", "--config", TRAIN_CONFIG, "--run-name", f"{{VERSION}}-train"], cwd=REPO_ROOT, env=ENV, check=True)\nsubprocess.run([sys.executable, "-m", "autoresearch_hv.hndsr_vr.evaluate_runner", "--config", TRAIN_CONFIG, "--run-name", f"{{VERSION}}-eval", "--checkpoint", f"artifacts/{{VERSION}}-train/checkpoints/{version}_train_best.pt"], cwd=REPO_ROOT, env=ENV, check=True)'''
        results_code = '''from pathlib import Path\nimport json\n\nmetrics_root = REPO_ROOT / "artifacts"\nfor candidate in [\n    metrics_root / f"{VERSION}-eval" / "metrics" / "eval_summary.json",\n    metrics_root / f"{VERSION}-smoke-eval" / "metrics" / "eval_summary.json",\n    metrics_root / f"{VERSION}-control" / "metrics" / "eval_summary.json",\n]:\n    if candidate.exists():\n        print(candidate)\n        print(json.dumps(json.loads(candidate.read_text(encoding="utf-8")), indent=2))\n'''
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["## Runtime Compatibility Check\n\nRun this notebook from the repo root or from a Kaggle working directory that contains the repo. The notebook is intentionally thin and delegates training, evaluation, and metrics work to repo modules.\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": setup_code.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Post-Restart GPU Sanity Check\n\nRun the next cell after any runtime restart. If CUDA is not available, keep the run diagnostic-focused and record the limitation in the handoff note.\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import torch\n", "print(f\"Torch: {torch.__version__}\")\n", "print(f\"CUDA available: {torch.cuda.is_available()}\")\n", "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": [f"# {version} HNDSR\n\n## Scratch SR3 Kaggle Baseline\n\nThis notebook is an execution shell around repo-owned code, not a notebook-only model implementation.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": [f"## Experiment Registry\n\n- Version: `{version}`\n- Parent: `{parent or 'none'}`\n- Lineage: `scratch`\n- Commands: `python -m autoresearch_hv validate-version --version {version}`, `python -m autoresearch_hv.hndsr_vr.train_runner --config`, `python -m autoresearch_hv.hndsr_vr.evaluate_runner --config`\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Paper Lineage and Hypothesis\n\nThe goal of `vR.1` is not to claim paper-quality output yet. The goal is to prove a traceable, repo-owned Kaggle loop that can later support deeper SR3 and HNDSR ablations.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": [f"## Dataset and Config Contract\n\n- Control config: `configs/hndsr_vr/{version}_control.yaml`\n- Smoke config: `configs/hndsr_vr/{version}_smoke.yaml`\n- Train config: `configs/hndsr_vr/{version}_train.yaml`\n- Fixed lane: Kaggle paired `4x` data\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Weights & Biases Setup\n\nW&B online logging is required for milestone completion. Configure `WANDB_API_KEY` in Kaggle Secrets before running the train/eval cells.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Training Execution\n\nRun the validation cell first, then the control evaluation, then smoke train/eval, then the full train/eval path.\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"subprocess.run([sys.executable, '-m', 'autoresearch_hv', 'validate-version', '--version', '{version}'], cwd=REPO_ROOT, env=ENV, check=True)\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_control.splitlines(True)},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_smoke.splitlines(True)},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run_train.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Evaluation and Export\n\nThe evaluation runner writes JSON summaries, sample grids, and tracker metadata under `artifacts/`. Pull the Kaggle bundle back before review.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Results Dashboard\n\nUse the next cell to inspect any evaluation summaries already produced by the run.\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": results_code.splitlines(True)},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Troubleshooting and Known Failure Modes\n\n- Missing W&B auth means the run is not milestone-complete.\n- CPU-only Kaggle runs are valid for diagnostics, not promotion.\n- Any Kaggle-side fix must be mirrored back into repo code or docs before the version is frozen.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Changelog\n\n- Initial scaffold for the immutable Kaggle baseline.\n"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Next Step Gate\n\nFreeze only after outputs are pulled, synced, reviewed, and mirrored. Fork the next version only from a written promotion decision.\n"]},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        return json.dumps(notebook, indent=1)

    def render_doc(self, version: str, parent: str | None, paths: VersionPaths) -> str:
        return f'''# {version} HNDSR\n\n## Objective\n\n`{version}` is the first immutable Kaggle notebook in the scratch lineage for the standalone AutoResearch repo. Its job is to prove the repo-owned lifecycle, not to overstate model quality.\n\n## Kaggle Run Guide\n\n1. Open `notebooks/versions/{paths.notebook.name}`.\n2. Run the runtime diagnostics cells first and confirm CUDA visibility if a GPU runtime is enabled.\n3. Run the validation cell before any training cell.\n4. Run the bicubic control evaluation to confirm the dataset and metrics path.\n5. Run the smoke training path to confirm checkpoint and evaluation wiring.\n6. Run the full training and full evaluation cells only after the smoke path succeeds.\n7. Pull the executed notebook and outputs back into the repo before review.\n\n## Config Contract\n\n- Full training config: `configs/hndsr_vr/{version}_train.yaml`\n- Smoke training config: `configs/hndsr_vr/{version}_smoke.yaml`\n- Bicubic control config: `configs/hndsr_vr/{version}_control.yaml`\n- Fixed scale: `4x`\n- W&B mode: `online` for milestone completion\n\n## Expected Artifacts\n\n- Control eval summary JSON under `artifacts/{version}-control/metrics/`\n- Smoke checkpoint and metrics under `artifacts/{version}-smoke/`\n- Full training checkpoint and metrics under `artifacts/{version}-train/`\n- Full evaluation summary and image strips under `artifacts/{version}-eval/`\n- Tracker records under each run\'s `tracker/` directory\n\n## Handoff Back For Review\n\nReturn all of the following after the Kaggle run:\n\n- The executed `{paths.notebook.name}`\n- Any Kaggle-side edits required for runtime stability\n- The best checkpoint path used for evaluation\n- The control, smoke, and full evaluation JSON summaries\n- The comparison grid image path\n- The W&B run URL\n- A short note about runtime duration, GPU type, and any failure modes hit during the run\n'''

    def render_review(self, version: str) -> str:
        return f'''# {version} HNDSR Review\n\n## Status\n\n- State: pending Kaggle run\n- Lineage: scratch (`vR.x`)\n- Scope: immutable SR3 Kaggle notebook for the standalone AutoResearch repo\n\n## Run Intake\n\n- Returned notebook path:\n- Kaggle runtime:\n- GPU or CPU:\n- Best checkpoint path:\n- Control summary path:\n- Smoke summary path:\n- Full evaluation summary path:\n- W&B URL:\n\n## Audit Checklist\n\n- Notebook structure stayed aligned with the contract.\n- Repo modules, not notebook cells, owned train/eval logic.\n- Metrics, checkpoint, and sample artifact paths are traceable.\n- W&B tracker state is explicit and reproducible.\n- Any Kaggle-only fixes were mirrored back into repo code or docs.\n\n## Findings\n\nPending execution. Populate this section with severity-ordered findings after the notebook comes back from Kaggle.\n\n## Roast\n\nPending execution. This section should be blunt, evidence-first, and specific about weak assumptions, wasted runtime, missing instrumentation, and misleading claims.\n\n## Promotion Decision\n\n- Decision:\n- Freeze current version, patch in place, or fork next version:\n- Rationale:\n'''

    def render_notebook_readme(self) -> str:
        return '''# Versioned Kaggle Notebooks\n\n- `vR.x_HNDSR.ipynb` is reserved for scratch-trained notebook versions.\n- `vR.P.x_HNDSR.ipynb` is reserved for externally pretrained notebook versions.\n- Do not overwrite a reviewed notebook version.\n- Pair every notebook with:\n  - `docs/notebooks/<stem>.md`\n  - `reports/reviews/<stem>.review.md`\n- Run `python -m autoresearch_hv --domain hndsr_vr validate-version --version <version>` before handing a notebook to Kaggle.\n'''

    def default_kernel_metadata(self) -> dict[str, Any]:
        return {
            "id": "INSERT_KAGGLE_USERNAME/vr1-hndsr",
            "title": "vR.1 HNDSR Scratch Baseline",
            "code_file": "vR.1_HNDSR.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": ["brsdincer/4x-satellite-image-super-resolution"],
            "competition_sources": [],
            "kernel_sources": [],
        }

    # -- Review / analysis ---------------------------------------------------

    def build_findings(
        self, version: str, manifest: dict[str, Any], eval_summary: dict[str, Any] | None, registry: dict[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, float]]:
        baseline = registry["control_baselines"]["kaggle_4x_bicubic_smoke"]
        findings: list[dict[str, str]] = []
        delta = {"psnr_delta": 0.0, "ssim_delta": 0.0}
        if not manifest.get("wandb_url"):
            findings.append({"severity": "high", "message": "W&B online URL is missing, so the phase-one milestone is not complete."})
        if not manifest.get("returned_notebook"):
            findings.append({"severity": "high", "message": "Executed notebook was not pulled back into the repo trace."})
        if not manifest.get("best_checkpoint"):
            findings.append({"severity": "high", "message": "Best checkpoint is missing from the synced Kaggle output bundle."})
        if not eval_summary:
            findings.append({"severity": "critical", "message": "No evaluation summary was found, so there is no trustworthy benchmark comparison."})
            return findings, delta
        delta["psnr_delta"] = round(float(eval_summary.get("psnr_mean", 0.0) - baseline["psnr_mean"]), 4)
        delta["ssim_delta"] = round(float(eval_summary.get("ssim_mean", 0.0) - baseline["ssim_mean"]), 4)
        if delta["psnr_delta"] <= 0:
            findings.append({"severity": "high", "message": f"PSNR is still below the bicubic control by {abs(delta['psnr_delta']):.4f}."})
        else:
            findings.append({"severity": "low", "message": f"PSNR exceeds the bicubic control by {delta['psnr_delta']:.4f}."})
        if delta["ssim_delta"] <= 0:
            findings.append({"severity": "medium", "message": f"SSIM is still below the bicubic control by {abs(delta['ssim_delta']):.4f}."})
        else:
            findings.append({"severity": "low", "message": f"SSIM exceeds the bicubic control by {delta['ssim_delta']:.4f}."})
        if not manifest.get("sample_grid"):
            findings.append({"severity": "medium", "message": "Qualitative sample grid is missing from the synced output bundle."})
        return findings, delta

    def ablation_suggestions(
        self, version: str, eval_summary: dict[str, Any] | None, delta: dict[str, float],
    ) -> list[str]:
        parts = version.split(".")
        if parts[:1] == ["vR"] and len(parts) == 2:
            major = f"vR.{int(parts[1]) + 1}"
            minor = f"{version}.1"
        elif parts[:1] == ["vR"] and len(parts) == 3:
            major = f"vR.{int(parts[1]) + 1}"
            minor = f"vR.{parts[1]}.{int(parts[2]) + 1}"
        else:
            major, minor = f"{version}-next-major", f"{version}.1"
        if not eval_summary:
            return [
                f"{minor}: fix missing evaluation export before any model change",
                f"{major}: rerun with explicit W&B online authentication and pulled notebook artifact",
            ]
        suggestions = [
            f"{minor}: increase model_channels from 32 to 64 while keeping the dataset and evaluation contract fixed",
            f"{major}: raise inference_steps from 10 to 20 without changing the training dataset or loss family",
        ]
        if delta.get("psnr_delta", 0.0) <= 0:
            suggestions.insert(0, f"{minor}: extend SR3 training epochs from 6 to 12 before changing architecture")
        return suggestions[:3]

    def roast_lines(self) -> list[str]:
        return [
            "The platform is not allowed to hide behind a pretty notebook if the pulled bundle is incomplete.",
            "Any run without an online W&B URL is operationally unfinished, not almost done.",
            "If bicubic still wins, architecture heroics are premature; the first fix is discipline, not bravado.",
        ]

    # -- Validation -----------------------------------------------------------

    def validate_version(self, version: str) -> list[str]:
        paths = self.resolve_version_paths(version)
        return validate_versioned_notebook(
            notebook_path=paths.notebook,
            doc_path=paths.doc,
            review_path=paths.review,
            full_config_path=paths.configs["train"],
            smoke_config_path=paths.configs["smoke"],
            control_config_path=paths.configs["control"],
            version=version,
        )
