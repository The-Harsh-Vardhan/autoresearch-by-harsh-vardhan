"""CLI entrypoint for AutoResearch by Harsh Vardhan."""

from __future__ import annotations

import argparse
import sys

from .core.domain_registry import discover_domains, get_domain
from .core.lifecycle import (
    kaggle_status,
    mirror_obsidian,
    next_ablation,
    pull_kaggle,
    push_kaggle,
    review_run,
    scaffold_version,
    sync_run,
    validate_version,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoResearch — general-purpose autonomous research platform",
    )
    parser.add_argument(
        "--domain",
        required=False,
        default=None,
        help="Research domain to operate on (required for most commands). Use 'list-domains' to see available domains.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Domain discovery commands -------------------------------------------

    subparsers.add_parser("list-domains", help="List all discovered research domains")

    domain_info = subparsers.add_parser("domain-info", help="Show details of a specific domain")
    domain_info.add_argument("--name", required=False, help="Domain name (uses --domain if not given)")

    # -- Lifecycle commands --------------------------------------------------

    scaffold = subparsers.add_parser("scaffold-version", help="Create notebook/doc/review/config assets")
    scaffold.add_argument("--version", required=True)
    scaffold.add_argument("--parent", default=None)
    scaffold.add_argument("--lineage", choices=["scratch", "pretrained"], default="scratch")
    scaffold.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate-version", help="Validate a version contract")
    validate.add_argument("--version", required=True)

    push = subparsers.add_parser("push-kaggle", help="Prepare metadata and push a notebook bundle")
    push.add_argument("--version", required=True)
    push.add_argument("--title", default=None)
    push.add_argument("--username", default=None)
    push.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("kaggle-status", help="Check Kaggle run status")
    status.add_argument("--version", required=True)
    status.add_argument("--username", default=None)
    status.add_argument("--dry-run", action="store_true")

    pull = subparsers.add_parser("pull-kaggle", help="Pull Kaggle outputs into artifacts")
    pull.add_argument("--version", required=True)
    pull.add_argument("--username", default=None)
    pull.add_argument("--dry-run", action="store_true")

    sync = subparsers.add_parser("sync-run", help="Index pulled Kaggle outputs into a run manifest")
    sync.add_argument("--version", required=True)
    sync.add_argument("--source-dir", default=None)
    sync.add_argument("--wandb-url", default=None)
    sync.add_argument("--dry-run", action="store_true")

    review = subparsers.add_parser("review-run", help="Generate a version review and roast")
    review.add_argument("--version", required=True)

    mirror = subparsers.add_parser("mirror-obsidian", help="Generate or write an Obsidian mirror note")
    mirror.add_argument("--version", required=True)
    mirror.add_argument("--output-dir", default=None)
    mirror.add_argument("--dry-run", action="store_true")

    ablation = subparsers.add_parser("next-ablation", help="Write the next bounded ablation suggestions")
    ablation.add_argument("--version", required=True)

    return parser


# Commands that do not require --domain
_DOMAIN_FREE_COMMANDS = {"list-domains"}


def _require_domain(args: argparse.Namespace) -> str:
    """Ensure --domain was provided for commands that need it."""
    if args.command in _DOMAIN_FREE_COMMANDS:
        return ""
    if args.command == "domain-info":
        return getattr(args, "name", None) or args.domain or ""
    if not args.domain:
        available = discover_domains()
        names = ", ".join(sorted(available)) or "(none found)"
        print(f"Error: --domain is required. Available domains: {names}", file=sys.stderr)
        raise SystemExit(1)
    return args.domain


def main() -> int:
    from .core.utils import load_dotenv
    load_dotenv()
    args = build_parser().parse_args()
    domain = _require_domain(args)

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
            print("Error: provide --domain or --name for domain-info.", file=sys.stderr)
            return 1
        manifest = get_domain(domain)
        print(f"Name:             {manifest.name}")
        print(f"Display Name:     {manifest.display_name}")
        print(f"Version Pattern:  {manifest.version_pattern}")
        print(f"Model Kinds:      {', '.join(manifest.model_kinds)}")
        print(f"Primary Metric:   {manifest.primary_metric} ({manifest.metric_direction})")
        print(f"Benchmark:        {manifest.benchmark_registry}")
        print(f"Config Dir:       {manifest.config_dir}")
        print(f"Programs Doc:     {manifest.programs_doc}")
        print(f"Entrypoints:")
        for key, val in manifest.entrypoints.items():
            print(f"  {key}: {val}")
        return 0

    if args.command == "scaffold-version":
        scaffold_version(domain, args.version, parent=args.parent, lineage=args.lineage, force=args.force)
    elif args.command == "validate-version":
        validate_version(domain, args.version)
    elif args.command == "push-kaggle":
        push_kaggle(domain, args.version, title=args.title, username=args.username, dry_run=args.dry_run)
    elif args.command == "kaggle-status":
        kaggle_status(domain, args.version, username=args.username, dry_run=args.dry_run)
    elif args.command == "pull-kaggle":
        pull_kaggle(domain, args.version, username=args.username, dry_run=args.dry_run)
    elif args.command == "sync-run":
        sync_run(domain, args.version, source_dir=args.source_dir, wandb_url=args.wandb_url, dry_run=args.dry_run)
    elif args.command == "review-run":
        review_run(domain, args.version)
    elif args.command == "mirror-obsidian":
        mirror_obsidian(domain, args.version, output_dir=args.output_dir, dry_run=args.dry_run)
    elif args.command == "next-ablation":
        next_ablation(domain, args.version)
    return 0
