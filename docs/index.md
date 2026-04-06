# Chakra CLI Docs

Chakra is a cyclic autonomous research system with a CLI that can scaffold experiments, run training and evaluation, validate version contracts, and orchestrate full research loops.

## What this site covers

- The Chakra CLI command surface
- Execution strategies for local, Kaggle, and automatic routing
- Lifecycle commands for scaffold, review, validate, and improve
- Troubleshooting and operational guidance

## Start here

1. Read the [CLI overview](cli/index.md) if you want the command map.
2. Open [Command Reference](cli/commands.md) for arguments and behavior.
3. Review [Execution](cli/execution.md) if you are running `run-execution` or Kaggle-backed jobs.
4. Use [Troubleshooting](cli/troubleshooting.md) for common failure modes.

## Recommended first command

```bash
python -m chakra list-domains
```

That confirms the package is installed and the repo's domain registry is discoverable.

## Core loop

Chakra's research lifecycle is:

```text
Sutra → Yantra → Rakshak → Vimarsh → Manthan
```

The docs in this site explain both the lifecycle and the CLI entry points that drive it.