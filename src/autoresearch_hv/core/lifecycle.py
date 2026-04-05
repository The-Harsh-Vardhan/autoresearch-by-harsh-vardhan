"""Generic lifecycle orchestration for all research domains."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None

from .domain_registry import DomainManifest, get_domain, load_lifecycle_hooks
from .interfaces import DomainLifecycleHooks, VersionPaths
from .utils import REPO_ROOT, read_json, write_json, write_text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        rendered = yaml.safe_dump(payload, sort_keys=False)
    else:
        rendered = json.dumps(payload, indent=2)
    path.write_text(rendered, encoding="utf-8")


def _resolve_kaggle_username(username: str | None, metadata: dict[str, Any]) -> str:
    if username:
        return username
    env_username = os.environ.get("KAGGLE_USERNAME")
    if env_username:
        return env_username
    kernel_id = metadata.get("id", "")
    if "/" in kernel_id and not kernel_id.startswith("INSERT_"):
        return kernel_id.split("/")[0]
    raise ValueError("Provide --username or set KAGGLE_USERNAME before pushing to Kaggle.")


def _run_cmd(args: list[str], dry_run: bool = False) -> None:
    print("$ " + " ".join(args))
    if dry_run:
        return
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def _find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _next_version_labels(version: str) -> tuple[str, str]:
    """Compute major+minor successor labels from a dot-separated version string.

    Works for any versioning scheme: vR.1 → (vR.2, vR.1.1), v1.0 → (v1.1, v1.0.1), etc.
    """
    parts = version.split(".")
    # Try incrementing the last numeric segment for major, append .1 for minor
    for i in range(len(parts) - 1, -1, -1):
        # Find the last part that ends with digits
        segment = parts[i]
        # Extract trailing digits from the segment
        j = len(segment)
        while j > 0 and segment[j - 1].isdigit():
            j -= 1
        if j < len(segment):
            prefix = segment[:j]
            num = int(segment[j:])
            major_parts = parts[:i] + [f"{prefix}{num + 1}"]
            major = ".".join(major_parts)
            minor = f"{version}.1"
            return major, minor
    # Fallback: can't parse version structure
    return f"{version}-next", f"{version}.1"


# ---------------------------------------------------------------------------
# Public lifecycle commands
# ---------------------------------------------------------------------------

def scaffold_version(domain_name: str, version: str, parent: str | None = None, lineage: str = "scratch", force: bool = False) -> None:
    """Create notebook, doc, review, and config assets for a version."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    configs = hooks.build_version_configs(version, parent=parent, lineage=lineage)

    critical_paths = [paths.notebook, paths.doc, paths.review] + list(paths.configs.values())
    for path in critical_paths:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing path: {path}")

    paths.notebook.parent.mkdir(parents=True, exist_ok=True)
    paths.doc.parent.mkdir(parents=True, exist_ok=True)
    paths.review.parent.mkdir(parents=True, exist_ok=True)
    paths.next_ablation.parent.mkdir(parents=True, exist_ok=True)
    paths.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.kaggle_output_dir.mkdir(parents=True, exist_ok=True)

    write_text(paths.notebook, hooks.render_notebook(version, parent, paths))
    write_text(paths.doc, hooks.render_doc(version, parent, paths))
    write_text(paths.review, hooks.render_review(version))
    write_text(paths.notebook_readme, hooks.render_notebook_readme())

    if not paths.kernel_metadata.exists() or force:
        write_json(paths.kernel_metadata, hooks.default_kernel_metadata())

    for config_key, config_payload in configs.items():
        config_path = paths.configs.get(config_key)
        if config_path:
            _write_yaml(config_path, config_payload)

    print(f"Scaffolded {version} assets for domain '{domain_name}'.")


def validate_version(domain_name: str, version: str) -> None:
    """Validate a version scaffold against the domain's contract."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    failures = hooks.validate_version(version)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print(f"{version} contract passed for domain '{domain_name}'.")


def push_kaggle(domain_name: str, version: str, title: str | None = None, username: str | None = None, dry_run: bool = False) -> None:
    """Prepare kernel metadata and push the versioned notebook to Kaggle."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    if not paths.notebook.exists():
        raise FileNotFoundError(paths.notebook)
    metadata = hooks.default_kernel_metadata()
    if paths.kernel_metadata.exists():
        metadata = read_json(paths.kernel_metadata)
    resolved_username = _resolve_kaggle_username(username, metadata)
    metadata["id"] = f"{resolved_username}/{hooks.version_slug(version)}"
    metadata["title"] = title or f"{version} {manifest.display_name}"
    metadata["code_file"] = paths.notebook.name
    if not dry_run:
        write_json(paths.kernel_metadata, metadata)
    print(json.dumps(metadata, indent=2))
    _run_cmd(["kaggle", "kernels", "push", "-p", str(paths.notebook.parent)], dry_run=dry_run)


def kaggle_status(domain_name: str, version: str, username: str | None = None, dry_run: bool = False) -> None:
    """Check Kaggle execution status for a version."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    metadata = read_json(paths.kernel_metadata)
    resolved_username = _resolve_kaggle_username(username, metadata)
    _run_cmd(["kaggle", "kernels", "status", f"{resolved_username}/{hooks.version_slug(version)}"], dry_run=dry_run)


def pull_kaggle(domain_name: str, version: str, username: str | None = None, dry_run: bool = False) -> None:
    """Pull Kaggle outputs for a version into the artifacts tree."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    metadata = read_json(paths.kernel_metadata)
    resolved_username = _resolve_kaggle_username(username, metadata)
    paths.kaggle_output_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd(["kaggle", "kernels", "output", f"{resolved_username}/{hooks.version_slug(version)}", "-p", str(paths.kaggle_output_dir)], dry_run=dry_run)


def sync_run(domain_name: str, version: str, source_dir: str | None = None, wandb_url: str | None = None, dry_run: bool = False) -> None:
    """Index pulled Kaggle outputs into a stable run manifest."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    source = Path(source_dir) if source_dir else paths.kaggle_output_dir
    if not source.exists():
        raise FileNotFoundError(source)
    returned_notebook = _find_first(source, [paths.notebook.name, "*.ipynb"])
    checkpoint = _find_first(source, ["*.pt"])
    eval_summary = _find_first(source, ["eval_summary.json"])
    train_summary = _find_first(source, ["train_summary.json"])
    sample_grid = _find_first(source, ["*grid*.png", "comparison_grid.png"])
    run_manifest = {
        "version": version,
        "domain": domain_name,
        "source_dir": str(source),
        "returned_notebook": str(returned_notebook) if returned_notebook else None,
        "best_checkpoint": str(checkpoint) if checkpoint else None,
        "eval_summary": str(eval_summary) if eval_summary else None,
        "train_summary": str(train_summary) if train_summary else None,
        "sample_grid": str(sample_grid) if sample_grid else None,
        "wandb_url": wandb_url,
    }
    print(json.dumps(run_manifest, indent=2))
    if not dry_run:
        write_json(paths.run_manifest, run_manifest)


def review_run(domain_name: str, version: str) -> None:
    """Generate a structured review and roast for a synced run."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)

    run_manifest = read_json(paths.run_manifest)
    registry = read_json(manifest.benchmark_registry) if manifest.benchmark_registry else {}
    eval_summary = read_json(run_manifest["eval_summary"]) if run_manifest.get("eval_summary") else None

    findings, delta = hooks.build_findings(version, run_manifest, eval_summary, registry)
    ablations = hooks.ablation_suggestions(version, eval_summary, delta)
    roast = hooks.roast_lines()

    has_critical = any(item["severity"] in {"critical", "high"} for item in findings)
    decision = "patch in place" if has_critical else "freeze and fork next version"

    findings_text = "\n".join(f"- [{f['severity'].upper()}] {f['message']}" for f in findings)
    roast_text = "\n".join(f"- {line}" for line in roast)
    ablation_text = "\n".join(f"- {item}" for item in ablations)

    review_text = "\n".join([
        f"# {version} {manifest.display_name} Review",
        "",
        "## Status",
        "",
        "- State: reviewed",
        f"- Domain: {manifest.display_name}",
        "",
        "## Run Intake",
        "",
        f"- Returned notebook path: {run_manifest.get('returned_notebook')}",
        f"- Best checkpoint path: {run_manifest.get('best_checkpoint')}",
        f"- Full evaluation summary path: {run_manifest.get('eval_summary')}",
        f"- W&B URL: {run_manifest.get('wandb_url')}",
        "",
        "## Findings",
        "",
        findings_text,
        "",
        "## Roast",
        "",
        roast_text,
        "",
        "## Promotion Decision",
        "",
        f"- Decision: {decision}",
        "- Next bounded ablations:",
        ablation_text,
        "",
    ])

    review_payload = {
        "version": version,
        "domain": domain_name,
        "decision": decision,
        "findings": findings,
        "roast": roast,
        "benchmark_delta": delta,
        "next_ablations": ablations,
        "wandb_url": run_manifest.get("wandb_url"),
        "eval_summary": eval_summary,
    }
    write_text(paths.review, review_text)
    write_json(paths.review_json, review_payload)
    write_text(paths.next_ablation, "# Next Ablations\n\n" + ablation_text + "\n")
    print(f"Wrote review for {version} ({domain_name}).")


def mirror_obsidian(domain_name: str, version: str, output_dir: str | None = None, dry_run: bool = False) -> None:
    """Generate a concise mirror note for Obsidian or a local output directory."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    run_manifest = read_json(paths.run_manifest)
    review_payload = read_json(paths.review_json)
    target_dir = Path(output_dir) if output_dir else REPO_ROOT / "artifacts" / "obsidian_mirror"
    note_path = target_dir / f"{paths.stem}.md"

    ablation_text = "\n".join(f"- {item}" for item in review_payload.get("next_ablations", []))
    note = "\n".join([
        f"# {version} Mirror",
        "",
        f"- Version: `{version}`",
        f"- Domain: `{domain_name}`",
        f"- W&B: {review_payload.get('wandb_url')}",
        f"- Kaggle output: `{run_manifest.get('source_dir')}`",
        f"- Returned notebook: `{run_manifest.get('returned_notebook')}`",
        f"- Decision: `{review_payload.get('decision')}`",
        "",
        "## Next Ablations",
        "",
        ablation_text,
        "",
    ])

    print(note)
    if not dry_run:
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(note, encoding="utf-8")
        print(f"Wrote mirror note to {note_path}")


def next_ablation(domain_name: str, version: str) -> None:
    """Write the current next-ablation suggestions for a version."""
    manifest = get_domain(domain_name)
    hooks = load_lifecycle_hooks(manifest)
    paths = hooks.resolve_version_paths(version)
    if paths.review_json.exists():
        payload = read_json(paths.review_json)
        suggestions = payload.get("next_ablations", [])
    else:
        run_manifest = read_json(paths.run_manifest)
        registry = read_json(manifest.benchmark_registry) if manifest.benchmark_registry else {}
        eval_summary = read_json(run_manifest["eval_summary"]) if run_manifest.get("eval_summary") else None
        _, delta = hooks.build_findings(version, run_manifest, eval_summary, registry)
        suggestions = hooks.ablation_suggestions(version, eval_summary, delta)
    content = "# Next Ablations\n\n" + "\n".join([f"- {item}" for item in suggestions]) + "\n"
    write_text(paths.next_ablation, content)
    print(content)
