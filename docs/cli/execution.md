# Execution Guide

Chakra's execution layer decides how a version should run based on the selected strategy and the current environment.

## Strategies

| Strategy | Behavior |
|---|---|
| `local` | Runs the version locally and keeps all work on the current machine. |
| `kaggle` | Pushes the notebook workflow to Kaggle for remote execution. |
| `auto` | Chooses the best available path using the execution engine's strategy rules. |

## What `run-execution` does

1. Loads the domain manifest and execution configuration.
2. Chooses a strategy.
3. Runs the selected execution path.
4. Optionally pulls Kaggle outputs back into the repo.

## Local execution

Local execution is the default when you want to keep everything on the same machine. It is the best choice for:

- quick iteration
- debugging runner behavior
- deterministic smoke checks

## Kaggle execution

Kaggle is useful when the domain is designed to run as a notebook workflow or when you want to use a managed remote environment.

The docs and the CLI keep the Kaggle path explicit so you can see when outputs are pushed or pulled.

## Automatic selection

Use `--strategy auto` when you want Chakra to choose the route. The execution engine considers the manifest, the runtime environment, and the local smoke gate before it commits to a remote run.

## Smoke gate

Chakra's lifecycle includes a local smoke gate before heavier remote work is attempted. That keeps the system from pushing an obviously broken version to Kaggle.

## Example

```bash
python -m chakra run-execution --domain tabular_cls --version v1.0 --strategy auto --pull-outputs
```

That is the simplest operator path for a full execution attempt.