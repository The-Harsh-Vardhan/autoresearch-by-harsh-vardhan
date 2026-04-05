"""Chakra CLI — the cyclic autonomous research interface.

Provides Chakra-named aliases for the research lifecycle:
    chakra sutra     → Plan: scaffold and freeze experiment configs
    chakra yantra    → Execute: train or evaluate a model
    chakra rakshak   → Guard: validate all version contracts
    chakra vimarsh   → Review: sync results and generate a review
    chakra manthan   → Improve: propose bounded ablation suggestions
    chakra aavart    → Full Cycle: run the complete Chakra loop end-to-end

All commands delegate to the same core lifecycle functions as the
traditional CLI — no logic is duplicated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .core.chakra_logger import ChakraLogger
from .core.domain_registry import discover_domains, get_domain, load_lifecycle_hooks
from .core.lifecycle import (
    next_ablation,
    review_run,
    scaffold_version,
    sync_run,
    validate_version,
)
from .core.utils import REPO_ROOT, load_dotenv


# ---------------------------------------------------------------------------
# Runner helpers — invoke domain train/eval as subprocesses
# ---------------------------------------------------------------------------

def _resolve_config_path(domain_name: str, version: str, variant: str) -> Path:
    """Resolve the config file path for a domain/version/variant."""
    return REPO_ROOT / "configs" / domain_name / f"{version}_{variant}.yaml"


def _find_checkpoint(version: str) -> Path | None:
    """Find the best checkpoint from a training run."""
    ckpt_dir = REPO_ROOT / "artifacts" / f"{version}-train" / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    return checkpoints[0] if checkpoints else None


def _run_train(domain_name: str, version: str, variant: str, device: str) -> None:
    """Run a domain's train_runner as a subprocess."""
    manifest = get_domain(domain_name)
    module = manifest.entrypoints["train_runner"]
    config_path = _resolve_config_path(domain_name, version, variant)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    run_name = f"{version}-{variant}"
    cmd = [
        sys.executable, "-m", module,
        "--config", str(config_path),
        "--run-name", run_name,
        "--device", device,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {run_name} (exit code {result.returncode})")


def _run_eval(domain_name: str, version: str, device: str) -> None:
    """Run a domain's evaluate_runner as a subprocess."""
    manifest = get_domain(domain_name)
    module = manifest.entrypoints["evaluate_runner"]
    config_path = _resolve_config_path(domain_name, version, "train")
    checkpoint = _find_checkpoint(version)
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found in artifacts/{version}-train/checkpoints/")
    run_name = f"{version}-eval"
    cmd = [
        sys.executable, "-m", module,
        "--config", str(config_path),
        "--run-name", run_name,
        "--checkpoint", str(checkpoint),
        "--device", device,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {run_name} (exit code {result.returncode})")


# ---------------------------------------------------------------------------
# Aavart — the full cycle orchestrator
# ---------------------------------------------------------------------------

def run_aavart(domain_name: str, version: str, device: str = "cpu", force: bool = False) -> None:
    """Execute a complete Chakra cycle (Aavart) for a domain and version."""
    log = ChakraLogger()
    log.aavart_start(domain_name, version)

    try:
        # 1. Sutra (Plan) — Scaffold
        log.sutra("Scaffolding version assets...")
        scaffold_version(domain_name, version, force=force)
        log.sutra("✓ Configs frozen")

        # 2. Yantra (Execute) — Control baseline
        control_config = _resolve_config_path(domain_name, version, "control")
        if control_config.exists():
            log.yantra("Running control baseline...")
            _run_train(domain_name, version, "control", device)
            log.yantra("✓ Control baseline complete")

        # 3. Yantra (Execute) — Smoke test
        smoke_config = _resolve_config_path(domain_name, version, "smoke")
        if smoke_config.exists():
            log.yantra("Running smoke test...")
            _run_train(domain_name, version, "smoke", device)
            log.yantra("✓ Smoke test complete")

        # 4. Yantra (Execute) — Full training
        log.yantra("Running full training...")
        _run_train(domain_name, version, "train", device)
        log.yantra("✓ Training complete")

        # 5. Yantra (Execute) — Evaluation
        log.yantra("Evaluating best checkpoint...")
        _run_eval(domain_name, version, device)
        log.yantra("✓ Evaluation complete")

        # 6. Vimarsh (Review) — Sync + Review
        log.vimarsh("Syncing results...")
        source_dir = str(REPO_ROOT / "artifacts" / f"{version}-train")
        sync_run(domain_name, version, source_dir=source_dir)
        log.vimarsh("Generating review...")
        review_run(domain_name, version)
        log.vimarsh("✓ Review written")

        # 7. Rakshak (Guard) — Validate
        log.rakshak("Validating version contract...")
        validate_version(domain_name, version)
        log.rakshak("✓ Contract passed")

        # 8. Manthan (Improve) — Ablation suggestions
        log.manthan("Generating ablation suggestions...")
        next_ablation(domain_name, version)
        log.manthan("✓ Ablations proposed")

        log.aavart_end(domain_name, version, "freeze and fork next version")

    except Exception as exc:
        stage = "unknown"
        # Determine which stage failed from the log context
        log.aavart_fail(domain_name, version, stage, str(exc))
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _add_domain_arg(parser: argparse.ArgumentParser) -> None:
    """Add --domain to a subcommand parser."""
    parser.add_argument("--domain", required=True,
                        help="Research domain to operate on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chakra",
        description=(
            "Chakra — Autonomous Research System\n\n"
            "A cyclic research engine: Plan → Execute → Guard → Review → Improve → Repeat\n\n"
            "Each command maps to a stage of the Chakra cycle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Sutra (Plan) -------------------------------------------------------
    sutra = subparsers.add_parser(
        "sutra",
        help="Sutra (Plan) — Create and freeze experiment plan",
        description="Sutra (सूत्र) means 'thread' or 'formula'. This stage scaffolds version assets "
                    "and freezes the experiment configuration — like weaving the plan before execution.",
    )
    _add_domain_arg(sutra)
    sutra.add_argument("--version", required=True, help="Version label (e.g., v1.0)")
    sutra.add_argument("--parent", default=None, help="Parent version to inherit from")
    sutra.add_argument("--lineage", choices=["scratch", "pretrained"], default="scratch")
    sutra.add_argument("--force", action="store_true", help="Overwrite existing assets")

    # -- Yantra (Execute) ----------------------------------------------------
    yantra = subparsers.add_parser(
        "yantra",
        help="Yantra (Execute) — Run training or evaluation",
        description="Yantra (यन्त्र) means 'instrument' or 'machine'. This stage executes "
                    "the training or evaluation pipeline — the engine that produces results.",
    )
    _add_domain_arg(yantra)
    yantra.add_argument("--version", required=True)
    yantra.add_argument("--stage", required=True, choices=["control", "smoke", "train", "eval"],
                        help="Which execution stage to run")
    yantra.add_argument("--device", default="cpu", help="Device: cpu or cuda")

    # -- Rakshak (Guard) -----------------------------------------------------
    rakshak = subparsers.add_parser(
        "rakshak",
        help="Rakshak (Guard) — Validate version contracts",
        description="Rakshak (रक्षक) means 'guardian'. This stage validates that all required "
                    "files exist and the version contract is satisfied — protecting research integrity.",
    )
    _add_domain_arg(rakshak)
    rakshak.add_argument("--version", required=True)

    # -- Vimarsh (Review) ----------------------------------------------------
    vimarsh = subparsers.add_parser(
        "vimarsh",
        help="Vimarsh (Review) — Sync results and generate review",
        description="Vimarsh (विमर्श) means 'reflection' or 'analysis'. This stage syncs training "
                    "outputs and generates a structured review with metric deltas and a roast.",
    )
    _add_domain_arg(vimarsh)
    vimarsh.add_argument("--version", required=True)
    vimarsh.add_argument("--source-dir", default=None, help="Override source directory for sync")

    # -- Manthan (Improve) ---------------------------------------------------
    manthan = subparsers.add_parser(
        "manthan",
        help="Manthan (Improve) — Generate ablation suggestions",
        description="Manthan (मन्थन) means 'churning' — like the mythological churning of the ocean. "
                    "This stage analyzes results and proposes bounded improvements for the next iteration.",
    )
    _add_domain_arg(manthan)
    manthan.add_argument("--version", required=True)

    # -- Aavart (Full Cycle) -------------------------------------------------
    aavart = subparsers.add_parser(
        "aavart",
        help="Aavart (Full Cycle) — Run the complete Chakra cycle end-to-end",
        description="Aavart (आवर्त) means 'cycle' or 'revolution'. This orchestrates the complete "
                    "loop: Sutra → Yantra → Rakshak → Vimarsh → Manthan in a single command.",
    )
    _add_domain_arg(aavart)
    aavart.add_argument("--version", required=True)
    aavart.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    aavart.add_argument("--force", action="store_true", help="Overwrite existing scaffolded assets")

    # -- Discovery commands --------------------------------------------------
    subparsers.add_parser("list-domains", help="List all discovered research domains")
    domain_info = subparsers.add_parser("domain-info", help="Show details of a specific domain")
    domain_info.add_argument("--name", required=False)

    return parser


_DOMAIN_FREE_COMMANDS = {"list-domains"}


def _require_domain(args: argparse.Namespace) -> str:
    if args.command in _DOMAIN_FREE_COMMANDS:
        return ""
    if args.command == "domain-info":
        return getattr(args, "name", None) or ""
    return getattr(args, "domain", "") or ""


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    domain = _require_domain(args)
    log = ChakraLogger()

    if args.command == "list-domains":
        domains = discover_domains()
        if not domains:
            print("No research domains discovered.")
        else:
            print(f"{'Name':<20} {'Display Name':<40} {'Primary Metric':<20}")
            print("-" * 80)
            for name, manifest in sorted(domains.items()):
                print(f"{name:<20} {manifest.display_name:<40} {manifest.primary_metric:<20}")
        return 0

    if args.command == "domain-info":
        if not domain:
            print("Error: provide --domain or --name.", file=sys.stderr)
            return 1
        manifest = get_domain(domain)
        print(f"Name:             {manifest.name}")
        print(f"Display Name:     {manifest.display_name}")
        print(f"Version Pattern:  {manifest.version_pattern}")
        print(f"Model Kinds:      {', '.join(manifest.model_kinds)}")
        print(f"Primary Metric:   {manifest.primary_metric} ({manifest.metric_direction})")
        return 0

    if args.command == "sutra":
        log.sutra("Scaffolding version assets...")
        scaffold_version(domain, args.version, parent=args.parent, lineage=args.lineage, force=args.force)
        log.sutra("✓ Configs frozen")

    elif args.command == "yantra":
        stage = args.stage
        if stage == "eval":
            log.yantra(f"Evaluating {args.version}...")
            _run_eval(domain, args.version, args.device)
            log.yantra("✓ Evaluation complete")
        else:
            log.yantra(f"Running {stage} for {args.version}...")
            _run_train(domain, args.version, stage, args.device)
            log.yantra(f"✓ {stage.capitalize()} complete")

    elif args.command == "rakshak":
        log.rakshak("Validating version contract...")
        validate_version(domain, args.version)
        log.rakshak("✓ Contract passed")

    elif args.command == "vimarsh":
        source_dir = args.source_dir or str(REPO_ROOT / "artifacts" / f"{args.version}-train")
        log.vimarsh("Syncing results...")
        sync_run(domain, args.version, source_dir=source_dir)
        log.vimarsh("Generating review...")
        review_run(domain, args.version)
        log.vimarsh("✓ Review written")

    elif args.command == "manthan":
        log.manthan("Generating ablation suggestions...")
        next_ablation(domain, args.version)
        log.manthan("✓ Ablations proposed")

    elif args.command == "aavart":
        run_aavart(domain, args.version, device=args.device, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
