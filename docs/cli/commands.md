# Command Reference

This page documents the commands exposed by the Chakra CLIs.

## Discovery

### `python -m chakra list-domains`

Lists all discovered research domains with their display names and primary metrics.

### `python -m chakra domain-info`

Shows the manifest metadata for a single domain.

Required argument:

- `--name` or `--domain`

## Planning

### `python -m chakra scaffold-version`

Creates the notebook, version docs, review template, and config files for a domain version.

Required arguments:

- `--version`

Optional arguments:

- `--parent`
- `--lineage {scratch,pretrained}`
- `--force`

### `chakra sutra`

Chakra-branded alias for the same plan-scaffolding step.

Required arguments:

- `--domain`
- `--version`

Optional arguments:

- `--parent`
- `--lineage {scratch,pretrained}`
- `--force`

## Execution

### `python -m chakra run-execution`

Runs a version using the configured strategy.

Optional strategy values:

- `local`
- `kaggle`
- `auto`

Useful flags:

- `--pull-outputs` to pull outputs after Kaggle completion
- `--dry-run` to print the decision flow without mutating state

### `chakra yantra`

Chakra-branded execution command.

Required arguments:

- `--domain`
- `--version`
- `--stage {control,smoke,train,eval}`

Optional arguments:

- `--device`

## Validation and review

### `python -m chakra validate-version`

Checks that the version contract is complete and that required files exist.

### `python -m chakra sync-run`

Indexes a run directory into the structured manifest used by reviews.

### `python -m chakra review-run`

Generates the human-readable review for a version.

### `python -m chakra next-ablation`

Writes bounded follow-up ablation ideas for the next iteration.

## Chakra aliases

| Alias | Meaning | Maps to |
|---|---|---|
| `chakra sutra` | Plan | Scaffold version assets |
| `chakra yantra` | Execute | Train or evaluate |
| `chakra rakshak` | Guard | Validate the version contract |
| `chakra vimarsh` | Review | Sync outputs and generate a review |
| `chakra manthan` | Improve | Generate ablation suggestions |
| `chakra aavart` | Full cycle | Run the complete lifecycle |