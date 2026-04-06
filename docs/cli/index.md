# Chakra CLI Overview

The Chakra CLI is the user-facing interface for the project. It exposes the lifecycle stages as commands so you can work either stage-by-stage or as a full cycle.

## Command families

| Family | Commands | Purpose |
|---|---|---|
| Discovery | `list-domains`, `domain-info` | Inspect the registered domains and their metadata. |
| Planning | `scaffold-version` / `sutra` | Create version assets and freeze the experiment contract. |
| Execution | `yantra`, `run-execution` | Train, evaluate, or route execution through local/Kaggle strategies. |
| Validation | `validate-version` / `rakshak` | Check that a version is complete and well-formed. |
| Review | `sync-run`, `review-run`, `vimarsh` | Index run outputs and generate a structured review. |
| Improvement | `next-ablation`, `manthan` | Produce bounded follow-up ideas for the next iteration. |

## Installation

```bash
pip install -e ".[dev]"
```

That installs the package itself plus the developer dependencies used for tests and docs work.

## Sanity check

```bash
python -m chakra list-domains
```

Expected output includes the shipped domains such as `hndsr_vr`, `nlp_lm`, and `tabular_cls`.

## Two CLI surfaces

Chakra keeps both entry points:

- `chakra ...` for the Chakra-branded lifecycle commands
- `python -m chakra ...` for the lower-level domain and execution commands

Use the Chakra CLI for the normal workflow. Use the Python module entry point when you need the broader domain lifecycle surface.