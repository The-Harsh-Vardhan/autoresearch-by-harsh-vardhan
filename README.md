# AutoResearch by Harsh Vardhan

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A general-purpose autonomous research platform for bounded, reviewable model-training experiments. Supports any research domain — computer vision, NLP, audio, tabular, reinforcement learning, and beyond — through a pluggable domain architecture.

## Architecture

```
core/                     Domain-agnostic engine
  ├── interfaces.py       Protocols every domain implements
  ├── domain_registry.py  Auto-discovers domains from domain.yaml manifests
  ├── lifecycle.py        Generic lifecycle (scaffold → train → review → promote)
  ├── tracker.py          W&B + null experiment tracker
  └── utils.py            Shared config loading, seeding, path helpers

domains/                  Pluggable research lanes
  ├── hndsr_vr/           CV: satellite super-resolution (SR3, PSNR ↑)
  └── nlp_lm/             NLP: character-level language model (GPT-nano, BPB ↓)
```

Each domain declares itself via a `domain.yaml` manifest and implements the `DomainLifecycleHooks` protocol. The CLI auto-discovers all domains — no manual registration.

## Shipped Domains

| Domain | Description | Primary Metric | Models |
|--------|-------------|---------------|--------|
| `hndsr_vr` | Satellite image super-resolution | PSNR ↑ | SR3 diffusion, bicubic baseline |
| `nlp_lm` | Character-level language modelling | BPB ↓ | GPT-nano transformer, bigram baseline |

## Project Layout

```
.
├── src/autoresearch_hv/
│   ├── core/                   Domain-agnostic engine
│   ├── domains/
│   │   ├── hndsr_vr/           CV domain (models, data, metrics, lifecycle)
│   │   └── nlp_lm/             NLP domain (models, data, metrics, lifecycle)
│   ├── hndsr_vr/               Backwards-compat shims → domains/hndsr_vr/
│   └── cli.py                  CLI with --domain dispatch
├── configs/
│   ├── hndsr_vr/               HNDSR config variants (base, control, smoke, train)
│   └── nlp_lm/                 NLP config variants
├── data/                       Sample datasets
├── benchmarks/                 Domain baseline registries (JSON)
├── programs/                   Human-owned research program docs
├── docs/                       Per-version run docs and audits
├── notebooks/versions/         Immutable Kaggle/Colab notebooks
├── reports/
│   ├── reviews/                Per-version audit/roast docs
│   └── generated/              Auto-generated ablation suggestions
├── tests/                      Test suite
├── pyproject.toml              Build config and dependencies
├── LICENSE                     MIT
└── CONTRIBUTING.md             Dev setup and contribution guide
```

## Quick Start

### Install

```bash
# Clone
git clone https://github.com/The-Harsh-Vardhan/autoresearch-by-harsh-vardhan.git
cd autoresearch-by-harsh-vardhan

# Install (into a venv recommended)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Explore

```bash
# List all available research domains
python -m autoresearch_hv list-domains

# Inspect a specific domain
python -m autoresearch_hv --domain hndsr_vr domain-info
python -m autoresearch_hv --domain nlp_lm domain-info
```

### Research Lifecycle

```bash
# Scaffold a new experiment version
python -m autoresearch_hv --domain hndsr_vr scaffold-version --version vR.2

# Validate the version contract (notebook, doc, review, configs all exist)
python -m autoresearch_hv --domain hndsr_vr validate-version --version vR.1

# After training on Kaggle, sync the results
python -m autoresearch_hv --domain hndsr_vr sync-run --version vR.1 --source-dir ./kaggle-output

# Generate a review/roast
python -m autoresearch_hv --domain hndsr_vr review-run --version vR.1

# Write an Obsidian mirror note
python -m autoresearch_hv --domain hndsr_vr mirror-obsidian --version vR.1
```

## CLI Reference

The primary entrypoint is `python -m autoresearch_hv`.

### Global Commands

| Command | Description |
|---------|-------------|
| `list-domains` | Show all discovered research domains |
| `domain-info` | Show details of a specific domain |

### Lifecycle Commands (require `--domain`)

| Command | Description |
|---------|-------------|
| `scaffold-version` | Create notebook, doc, review, and config assets |
| `validate-version` | Validate a version's contract (all files exist) |
| `push-kaggle` | Prepare and push a notebook to Kaggle |
| `kaggle-status` | Check Kaggle kernel run status |
| `pull-kaggle` | Pull Kaggle outputs into artifacts |
| `sync-run` | Index pulled outputs into a run manifest |
| `review-run` | Generate a version review and roast |
| `mirror-obsidian` | Generate an Obsidian mirror note |
| `next-ablation` | Write next bounded ablation suggestions |

## Adding a New Domain

1. Create `src/autoresearch_hv/domains/<name>/`
2. Add a `domain.yaml` manifest — see [hndsr_vr/domain.yaml](src/autoresearch_hv/domains/hndsr_vr/domain.yaml) for format
3. Implement `LifecycleHooks` class in `lifecycle.py` (follows the `DomainLifecycleHooks` protocol)
4. Add `models.py`, `dataset.py`, `metrics.py`, `train_runner.py`, `evaluate_runner.py`
5. Create configs under `configs/<name>/`
6. Create `benchmarks/<name>_registry.json` and `programs/<name>.md`
7. Add tests under `tests/test_<name>.py`
8. Done — the CLI discovers your domain automatically

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development workflow.

## Testing

```bash
python -m pytest tests/ -v
```

Tests requiring PyTorch are **skipped automatically** when it's not installed. Core infrastructure (domain registry, CLI) tests always run.

## License

[MIT](LICENSE)
