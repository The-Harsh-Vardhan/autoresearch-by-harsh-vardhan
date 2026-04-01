"""CLI entrypoint for AutoResearch by Harsh Vardhan."""

from __future__ import annotations

import argparse

from .hndsr_vr.lifecycle import (
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
    parser = argparse.ArgumentParser(description="AutoResearch by Harsh Vardhan")
    subparsers = parser.add_subparsers(dest="command", required=True)

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


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "scaffold-version":
        scaffold_version(args.version, parent=args.parent, lineage=args.lineage, force=args.force)
    elif args.command == "validate-version":
        validate_version(args.version)
    elif args.command == "push-kaggle":
        push_kaggle(args.version, title=args.title, username=args.username, dry_run=args.dry_run)
    elif args.command == "kaggle-status":
        kaggle_status(args.version, username=args.username, dry_run=args.dry_run)
    elif args.command == "pull-kaggle":
        pull_kaggle(args.version, username=args.username, dry_run=args.dry_run)
    elif args.command == "sync-run":
        sync_run(args.version, source_dir=args.source_dir, wandb_url=args.wandb_url, dry_run=args.dry_run)
    elif args.command == "review-run":
        review_run(args.version)
    elif args.command == "mirror-obsidian":
        mirror_obsidian(args.version, output_dir=args.output_dir, dry_run=args.dry_run)
    elif args.command == "next-ablation":
        next_ablation(args.version)
    return 0
