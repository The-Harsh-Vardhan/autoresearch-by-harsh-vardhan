<p align="center">
  <img src="CHAKRA%20Logo.png" alt="Chakra Logo" width="200"/>
</p>

# Chakra — Autonomous Research System

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![W&B Tracking](https://img.shields.io/badge/W%26B-Tracked-orange.svg)](https://wandb.ai)
[![47 Tests Passed](https://img.shields.io/badge/Tests-47%20Passed-brightgreen.svg)](#testing)
[![3 Domains](https://img.shields.io/badge/Domains-3%20Shipped-purple.svg)](#shipped-domains)

**A cyclic, autonomous research engine that plans, trains, guards, reviews, and improves ML experiments — then repeats.**

Chakra eliminates manual experiment management. You define a research domain once. Chakra handles the rest: scaffolding configs, running baselines, training models, evaluating checkpoints, validating contracts, generating reviews, and proposing the next iteration. One command runs the entire loop.

```bash
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

This single command executes the full research cycle — from creating the experiment plan to proposing improvements for v1.1.

---

## The Cycle

Every experiment in Chakra follows a five-stage loop:

```
    ┌──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
  Plan ──→ Execute ──→ Guard ──→ Review ──→ Improve
    │                                          │
    └──────────────────────────────────────────┘
```

| # | Stage | What Happens |
|---|-------|-------------|
| 1 | **Plan** | Scaffold version assets. Freeze configs. Define the hypothesis. |
| 2 | **Execute** | Train control baseline → smoke test → full training → evaluate. |
| 3 | **Guard** | Validate that all required files and contracts are satisfied. |
| 4 | **Review** | Sync results. Generate metric deltas. Produce a structured roast. |
| 5 | **Improve** | Analyze findings. Propose bounded ablation suggestions for next version. |

When the cycle completes, the system either **freezes** the version (if results are good) or **forks** a new version with proposed improvements. Then the cycle repeats.

### Chakra Identity

Each stage has a Sanskrit name that reflects its purpose:

| Chakra Term | Meaning | Stage |
|-------------|---------|-------|
| **Sutra** (सूत्र) | Thread / Formula | Plan |
| **Yantra** (यन्त्र) | Instrument / Machine | Execute |
| **Rakshak** (रक्षक) | Guardian | Guard |
| **Vimarsh** (विमर्श) | Reflection / Analysis | Review |
| **Manthan** (मन्थन) | Churning (of the ocean) | Improve |
| **Aavart** (आवर्त) | Cycle / Revolution | Full Loop |

> *Chakra* (चक्र) means *wheel* — the cycle that never stops turning.

---

## Shipped Domains & Real Results

| Domain | Task | Metric | Control Baseline | Trained Model | Δ |
|--------|------|--------|-----------------|--------------|---|
| **`hndsr_vr`** | Satellite super-resolution | PSNR ↑ | Bicubic baseline | SR3 diffusion | — |
| **`nlp_lm`** | Character-level language model | BPB ↓ | Bigram: **6.38** | GPT-nano: **3.59** | **44% ↓** |
| **`tabular_cls`** (Iris) | Flower classification | Accuracy ↑ | Logistic: **16.7%** | MLP: **93.3%** | **+76.7pp** |
| **`tabular_cls`** (Titanic) | Survival prediction | Accuracy ↑ | Logistic: **58.3%** | MLP: **83.6%** | **+25.3pp** |

> All results above are real, measured outputs from runs executed during development — not estimates.

---

## Quick Start

> 📖 **For the full guide** see [How to Use Chakra](docs/how_to_use.md) — covers every command, config option, W&B setup, troubleshooting, and how to add your own domain. For a 5-minute walkthrough, see the [Quickstart](docs/quickstart.md).

### 1. Install

```bash
git clone https://github.com/The-Harsh-Vardhan/autoresearch-by-harsh-vardhan.git
cd autoresearch-by-harsh-vardhan
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev]"
```

### 2. Run the Full Cycle (One Command)

```bash
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

This runs the complete Aavart (Full Cycle):

```
🔁 [Chakra] Starting Aavart (Full Cycle) — tabular_cls v1.0
  📜 Sutra (Plan): Scaffolding version assets...
  📜 Sutra (Plan): ✓ Configs frozen
  ⚙️ Yantra (Execute): Running control baseline...
  ⚙️ Yantra (Execute): ✓ Control baseline complete
  ⚙️ Yantra (Execute): Running smoke test...
  ⚙️ Yantra (Execute): ✓ Smoke test complete
  ⚙️ Yantra (Execute): Running full training...
  ⚙️ Yantra (Execute): ✓ Training complete
  ⚙️ Yantra (Execute): Evaluating best checkpoint...
  ⚙️ Yantra (Execute): ✓ Evaluation complete
  🔍 Vimarsh (Review): Syncing results...
  🔍 Vimarsh (Review): ✓ Review written
  🛡️ Rakshak (Guard): Validating version contract...
  🛡️ Rakshak (Guard): ✓ Contract passed
  🔄 Manthan (Improve): Generating ablation suggestions...
  🔄 Manthan (Improve): ✓ Ablations proposed
✅ [Chakra] Aavart complete — tabular_cls v1.0. Decision: freeze and fork next version.
```

### 3. Or Use Individual Stages

```bash
# Plan
chakra sutra --domain tabular_cls --version v1.0 --force

# Execute
chakra yantra --domain tabular_cls --version v1.0 --stage train --device cpu

# Guard
chakra rakshak --domain tabular_cls --version v1.0

# Review
chakra vimarsh --domain tabular_cls --version v1.0

# Improve
chakra manthan --domain tabular_cls --version v1.0
```

### 4. Traditional CLI (Still Works)

All original commands remain available through `python -m chakra`:

```bash
python -m chakra list-domains
python -m chakra --domain tabular_cls scaffold-version --version v1.0 --force
python -m chakra --domain tabular_cls validate-version --version v1.0
```

---

## Architecture

```
chakra/
├── core/                           # Domain-agnostic engine (Chakra kernel)
│   ├── interfaces.py               # DomainLifecycleHooks protocol
│   ├── domain_registry.py          # Auto-discovers domains from domain.yaml
│   ├── lifecycle.py                 # Generic scaffold → sync → review → promote
│   ├── chakra_logger.py            # Structured stage-aware logging
│   ├── tracker.py                  # W&B tracker + NullTracker fallback
│   └── utils.py                    # Config loading, seeding, path helpers
│
├── domains/                        # Each domain is a self-contained plugin
│   ├── hndsr_vr/                   # Satellite image super-resolution
│   ├── nlp_lm/                     # Character-level language model
│   └── tabular_cls/                # Tabular classification (Iris, Titanic)
│
├── cli.py                          # Traditional CLI (python -m chakra)
└── chakra_cli.py                   # Chakra CLI (chakra sutra/yantra/...)
```

### Chakra ↔ System Mapping

| Subsystem | Chakra Role | Code |
|-----------|-------------|------|
| `core/lifecycle.py` | Orchestrates the cycle | `scaffold_version`, `sync_run`, `review_run` |
| `core/chakra_logger.py` | Emits stage-aware logs | `ChakraLogger` |
| `core/tracker.py` | Records telemetry | `WandbTracker`, `NullTracker` |
| `core/domain_registry.py` | Discovers research lanes | `discover_domains()` |
| `core/interfaces.py` | Defines the domain contract | `DomainLifecycleHooks` |
| `domains/*/train_runner.py` | Yantra (Execute) — training | Per-domain subprocess |
| `domains/*/evaluate_runner.py` | Yantra (Execute) — evaluation | Per-domain subprocess |
| `chakra_cli.py` | Entry point for the Chakra interface | `run_aavart()` |

---

## CLI Reference

### Chakra Commands

| Command | Stage | Description |
|---------|-------|-------------|
| `chakra sutra` | Plan | Create and freeze experiment plan (scaffold assets + configs) |
| `chakra yantra` | Execute | Run training or evaluation (`--stage control\|smoke\|train\|eval`) |
| `chakra rakshak` | Guard | Validate that all version files and contracts exist |
| `chakra vimarsh` | Review | Sync training results and generate structured review |
| `chakra manthan` | Improve | Propose bounded ablation suggestions for next iteration |
| `chakra aavart` | Full Cycle | Run the complete Plan → Execute → Guard → Review → Improve loop |
| `chakra list-domains` | Discovery | List all auto-discovered research domains |

### Traditional Commands

| Command | Description |
|---------|-------------|
| `python -m chakra list-domains` | List all domains |
| `python -m chakra --domain D scaffold-version --version V` | Scaffold version assets |
| `python -m chakra --domain D validate-version --version V` | Validate version contract |
| `python -m chakra --domain D sync-run --version V` | Index results into manifest |
| `python -m chakra --domain D review-run --version V` | Generate review and roast |
| `python -m chakra --domain D next-ablation --version V` | Write ablation suggestions |
| `python -m chakra --domain D push-kaggle --version V` | Push notebook to Kaggle |
| `python -m chakra --domain D pull-kaggle --version V` | Pull Kaggle outputs |

---

## W&B Experiment Tracking

```bash
# Create .env in repo root
echo WANDB_API_KEY=your_key_here > .env
```

When a key is present, all runners stream metrics to W&B automatically. Without it, everything still works — metrics save to local JSON via `NullTracker`.

---

## Configuration System

Configs use YAML with an `inherits:` key for layered configuration:

```yaml
# configs/tabular_cls/v1.0_train.yaml
inherits: configs/tabular_cls/base.yaml

project:
  group: v1.0-train

training:
  epochs: 30
  checkpoint_name: v1.0_train_best.pt
```

Each version always has three config variants:

| Variant | Purpose |
|---------|---------|
| `*_control.yaml` | Baseline model (establishes the floor) |
| `*_smoke.yaml` | Quick pipeline sanity check (3 epochs, 5 batches) |
| `*_train.yaml` | Full training run |

---

## Adding a New Domain

Chakra is designed for zero-code-change domain addition:

```
src/chakra/domains/my_domain/
├── __init__.py
├── domain.yaml          # Domain manifest (name, metrics, entrypoints)
├── lifecycle.py          # Implements DomainLifecycleHooks protocol
├── models.py             # Domain-specific models
├── dataset.py            # Data loading and preprocessing
├── metrics.py            # Evaluation metrics
├── train_runner.py       # Training script with W&B tracking
└── evaluate_runner.py    # Evaluation script with W&B tracking
```

Register in `pyproject.toml`:
```toml
"chakra.domains.my_domain" = ["domain.yaml"]
```

Then:
```bash
chakra list-domains    # Your domain appears automatically
chakra aavart --domain my_domain --version v1.0 --device cpu
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full tutorial.

---

## Testing

```bash
python -m pytest tests/ -v
```

47 tests across all domains and core infrastructure:

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_core.py` | 16 | Config, utils, seeding, registry |
| `test_tabular_domain.py` | 9 | Discovery, protocol, models, dataset, metrics |
| `test_nlp_domain.py` | 6 | GPT-nano, bigram, dataset, metrics |
| `test_domain_registry.py` | 5 | Multi-domain discovery, manifests |
| `test_runtime_contract.py` | 6 | Path resolution, workspace isolation |
| `test_cli_dispatch.py` | 3 | CLI argument parsing |
| `test_lifecycle_review.py` | 1 | Full sync → review pipeline |
| `test_notebook_contract.py` | 1 | Notebook JSON structure |

---

## Inspirations

- **Chakra** (चक्र) — The wheel that represents cyclical, self-sustaining motion
- **[Sakana AI's AI Scientist](https://github.com/sakanaai/ai-scientist)** — Automated scientific discovery
- **[AutoKaggle](https://github.com/multimodal-art-projection/AutoKaggle)** — Multi-agent Kaggle orchestration
- **[W&B Experiment Tracking](https://wandb.ai/)** — MLOps telemetry backbone

---

## License

[MIT](LICENSE) — Use it, fork it, extend it.
