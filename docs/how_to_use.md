# How to Use Chakra — Complete Guide

This guide covers everything you need to know to use Chakra: from installation to running experiments, understanding results, tracking with W&B, extending the framework, and troubleshooting.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Project Structure](#2-project-structure)
3. [Core Concepts](#3-core-concepts)
4. [Running Your First Experiment](#4-running-your-first-experiment)
5. [The Seven-Step Lifecycle](#5-the-seven-step-lifecycle)
6. [Configuration System](#6-configuration-system)
7. [Weights & Biases Integration](#7-weights-biases-integration)
8. [CLI Reference](#8-cli-reference)
9. [Working with Each Domain](#9-working-with-each-domain)
10. [Adding Your Own Domain](#10-adding-your-own-domain)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Installation

### Requirements

- **Python 3.10+** (tested with 3.10, 3.11, 3.12, 3.13)
- **Git**
- **(Optional)** [Weights & Biases](https://wandb.ai/) account for cloud experiment tracking
- **(Optional)** [Kaggle CLI](https://github.com/Kaggle/kaggle-api) for cloud notebook execution

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/The-Harsh-Vardhan/Chakra-Autonomous-Research-System.git
cd Chakra-Autonomous-Research-System

# Create a virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\activate
# Windows Command Prompt:
.venv\Scripts\activate.bat

# Install with all dependencies (including dev/test tools)
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -m chakra list-domains
```

You should see output like:

```
Name                 Display Name                             Primary Metric
--------------------------------------------------------------------------------
hndsr_vr             HNDSR Satellite Super-Resolution         psnr_mean
nlp_lm               NLP Language Modelling                   val_bpb
tabular_cls          Tabular Classification                   accuracy
```

If you see all three domains, you're ready to go.

---

## 2. Project Structure

```
Chakra-Autonomous-Research-System/
│
├── src/chakra/                        # Python source code
│   ├── core/                          # Domain-agnostic engine (never touch for new domains)
│   │   ├── interfaces.py              # DomainLifecycleHooks protocol
│   │   ├── domain_registry.py         # Auto-discovers domains
│   │   ├── lifecycle.py               # Generic scaffold → sync → review → promote
│   │   ├── tracker.py                 # W&B tracker + NullTracker fallback
│   │   └── utils.py                   # Config loading, .env, seeding, path helpers
│   │
│   ├── domains/                       # Each domain is a self-contained plugin
│   │   ├── hndsr_vr/                  # Satellite super-resolution (CV)
│   │   ├── nlp_lm/                    # Character-level language model (NLP)
│   │   └── tabular_cls/               # Tabular classification (ML)
│   │
│    └── cli.py                         # Traditional CLI (python -m chakra)
│   └── chakra_cli.py                  # Chakra CLI (chakra sutra/yantra/...)
│
├── configs/                           # YAML configuration files
│   ├── hndsr_vr/                      # Per-domain config sets
│   ├── nlp_lm/
│   └── tabular_cls/
│
├── benchmarks/                        # Measured baseline registries (JSON)
├── data/                              # Datasets (e.g., titanic.csv)
├── programs/                          # Research program documents
├── docs/                              # Documentation
├── notebooks/versions/                # Generated Kaggle/Colab notebooks
├── reports/                           # Reviews and generated ablation plans
├── artifacts/                         # Training outputs (gitignored)
├── tests/                             # Test suite (47 tests)
├── .github/workflows/test.yml         # CI pipeline
├── pyproject.toml                     # Build config and dependencies
└── .env                               # W&B credentials (gitignored, create manually)
```

### Key Directories

| Directory | What Goes Here | Tracked in Git? |
|-----------|---------------|-----------------|
| `configs/` | YAML training configs | ✅ Yes |
| `artifacts/` | Checkpoints, metrics, manifests | ❌ No (gitignored) |
| `benchmarks/` | Measured baseline JSON registries | ✅ Yes |
| `notebooks/versions/` | Generated Kaggle notebooks | ✅ Yes |
| `reports/reviews/` | Version review/roast docs | ✅ Yes |
| `reports/generated/` | Ablation suggestions | ❌ No (gitignored) |

---

## 3. Core Concepts

### Domains

A **domain** is a self-contained research lane (e.g., NLP, computer vision, tabular ML). Each domain defines its own models, datasets, metrics, and lifecycle hooks. The core engine doesn't know or care about domain-specific details — it only calls the standardized protocol.

### Versions

Every experiment run is tied to a **version** (e.g., `v1.0`, `v2.0`, `v1.0.1`). Each version has:
- A **notebook** (Kaggle-ready `.ipynb`)
- A **doc** (markdown describing the experiment)
- A **review** (findings + roast + ablation suggestions)
- Three **configs**: `control`, `smoke`, `train`

### Config Variants

Each version always has three config variants:

| Variant | Purpose | When to Use |
|---------|---------|-------------|
| `control` | Baseline model (e.g., logistic regression), minimal or no training | First — establishes the accuracy floor |
| `smoke` | Main model with heavily limited batches (3 epochs, 5 batches) | Second — validates the pipeline runs end-to-end |
| `train` | Main model, full training (30+ epochs, all data) | Third — produces the real results |

### The Lifecycle

```
Scaffold → Control → Smoke → Train → Evaluate → Sync → Review → Validate → (Promote or Iterate)
```

### Frozen Ablation Plans

Once a version is scaffolded, its configs are **frozen**. This prevents goal-drift during autonomous runs. If you want to try different hyperparameters, create a new version (e.g., `v1.1`).

---

## 4. Running Your First Experiment

The fastest experiment uses the **Tabular Classification** domain with the Iris dataset (150 samples, runs in ~10 seconds on CPU).

### Step 1: Scaffold

```bash
python -m chakra --domain tabular_cls scaffold-version --version v1.0 --force
```

This generates the notebook, doc, review template, and config files for `v1.0`.

### Step 2: Control Baseline

```bash
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v1.0_control.yaml \
  --run-name v1.0-control \
  --device cpu
```

**Output:** `Best val accuracy: ~16.67%` — an untrained logistic regression on 3 classes is essentially random.

### Step 3: Smoke Test

```bash
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v1.0_smoke.yaml \
  --run-name v1.0-smoke \
  --device cpu
```

**Output:** `Best val accuracy: ~83%` — confirms the MLP is learning, pipeline is working.

### Step 4: Full Training

```bash
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v1.0_train.yaml \
  --run-name v1.0-train \
  --device cpu
```

**Output:** `Best val accuracy: ~93.3%` — the MLP learns the Iris classification task.

### Step 5: Evaluate

```bash
python -m chakra.domains.tabular_cls.evaluate_runner \
  --config configs/tabular_cls/v1.0_train.yaml \
  --run-name v1.0-eval \
  --checkpoint artifacts/v1.0-train/checkpoints/v1.0_train_best.pt \
  --device cpu
```

**Output:** `Accuracy=93.33%, F1=0.9296`

### Step 6: Sync Results

```bash
python -m chakra --domain tabular_cls sync-run \
  --version v1.0 \
  --source-dir artifacts/v1.0-train
```

This indexes training outputs into a structured run manifest at `artifacts/runs/v1.0/run_manifest.json`.

### Step 7: Review

```bash
python -m chakra --domain tabular_cls review-run --version v1.0
```

Generates findings, metric deltas against the benchmark baseline, ablation suggestions, and a "roast" at `reports/reviews/v1.0_Tabular_CLS.review.md`.

### Step 8: Validate

```bash
python -m chakra --domain tabular_cls validate-version --version v1.0
```

**Output:** `v1.0 contract passed for domain 'tabular_cls'.`

This checks that all required files (notebook, doc, review, configs) exist.

---

## 5. The Seven-Step Lifecycle

Every version goes through this lifecycle, regardless of domain:

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐
│ Scaffold │→ │ Control │→ │ Smoke   │→ │ Train    │
└─────────┘  └─────────┘  └─────────┘  └──────────┘
                                              │
     ┌──────────┐  ┌────────┐  ┌──────────┐   │
     │ Validate │← │ Review │← │   Sync   │ ←─┘
     └──────────┘  └────────┘  └──────────┘
           │
     ┌─────┴─────┐
     │  Promote?  │
     └─────┬─────┘
      Yes  │  No
       │   │
    Freeze Fork→ v1.1
```

| Step | CLI Command | What It Does |
|------|------------|--------------|
| **Scaffold** | `scaffold-version --version v1.0` | Creates notebook, doc, review, configs |
| **Control** | `train_runner --config *_control.yaml` | Runs baseline model (sets the floor) |
| **Smoke** | `train_runner --config *_smoke.yaml` | Quick pipeline sanity check |
| **Train** | `train_runner --config *_train.yaml` | Full training run |
| **Evaluate** | `evaluate_runner --checkpoint ...` | Evaluate on validation set |
| **Sync** | `sync-run --version v1.0 --source-dir ...` | Index outputs into a run manifest |
| **Review** | `review-run --version v1.0` | Generate findings + ablation suggestions |
| **Validate** | `validate-version --version v1.0` | Check all files exist |

---

## 6. Configuration System

### YAML with Inheritance

All configs use YAML format with an `inherits:` key for layered configuration:

```yaml
# configs/tabular_cls/v1.0_train.yaml
inherits: configs/tabular_cls/base.yaml    # ← loads base config first

project:
  group: v1.0-train                         # ← only override what changes

training:
  epochs: 30
  checkpoint_name: v1.0_train_best.pt
```

The `inherits` key tells the config loader to:
1. Load `base.yaml` first
2. Deep-merge the current file's keys on top
3. Return the merged result

This means you only need to specify what's **different** from the base.

### Config Schema

Here's every field supported in a config file:

```yaml
seed: 42                              # Random seed for reproducibility

project:
  name: chakra                        # W&B project name
  group: v1.0-train                    # W&B run group
  tags: [tabular, classification]      # W&B tags

runtime:
  version: v1.0                        # Version label
  lineage: scratch                     # "scratch" or "pretrained"
  parent: null                         # Parent version (null for first version)

paths:
  artifact_root: artifacts             # Where checkpoints/metrics are saved
  report_root: reports                 # Where reviews/generated docs go

data:
  dataset: iris                        # Dataset name ("iris" or "titanic")
  data_file: data/titanic.csv          # Path to CSV (Titanic only)
  val_split: 0.2                       # Validation set fraction
  batch_size: 32                       # DataLoader batch size

tracking:
  enabled: true                        # Enable W&B tracking
  mode: online                         # "online", "offline", or "disabled"
  project: chakra
  entity: null                         # W&B team/entity (null for personal)
  notes: "Description of this run"     # Shown in W&B dashboard

model:
  kind: mlp                            # Model type ("logistic", "mlp", etc.)
  hidden_dim: 64                       # MLP hidden layer size
  dropout: 0.2                         # Dropout rate

training:
  epochs: 30                           # Number of training epochs
  lr: 0.001                            # Learning rate
  weight_decay: 0.0001                 # L2 regularization
  max_train_batches: null              # Limit train batches (null = all)
  max_val_batches: null                # Limit val batches (null = all)
  checkpoint_name: v1.0_train_best.pt  # Best checkpoint filename

evaluation:
  sample_limit: 100                    # Max val batches during evaluation
  save_limit: 8                        # Max samples to save for visualization
```

### Creating a New Version's Configs

For a new version (e.g., `v1.1`), create three config files:

```
configs/tabular_cls/v1.1_control.yaml   # Baseline
configs/tabular_cls/v1.1_smoke.yaml     # Quick check
configs/tabular_cls/v1.1_train.yaml     # Full training
```

Each inherits from `base.yaml` and overrides the specific fields for that variant.

---

## 7. Weights & Biases Integration

### Setup

1. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

2. Create a `.env` file in the project root:
   ```
   WANDB_API_KEY=your_api_key_here
   ```

3. That's it. All runners call `load_dotenv()` at startup, which loads the key.

### How It Works

The `init_tracker()` function in `core/tracker.py` decides which backend to use:

```
Has WANDB_API_KEY? ──Yes──→ Has wandb installed? ──Yes──→ WandbTracker (full cloud tracking)
       │                          │
       No                         No
       │                          │
       └──────────────────────────┴──→ NullTracker (local JSON only)
```

**You never lose data.** If W&B isn't available, the `NullTracker` saves all metrics and artifacts to local JSON files in the `artifacts/` directory.

### Tracking Modes

Set in your config YAML under `tracking.mode`:

| Mode | Behavior |
|------|----------|
| `online` | Streams metrics to W&B cloud in real-time |
| `offline` | Saves W&B run locally, sync later with `wandb sync` |
| `disabled` | Forces NullTracker regardless of API key |

### What Gets Tracked

| Artifact | Logged As |
|----------|-----------|
| Full resolved config | W&B artifact (type: `config`) |
| Dataset split manifest | W&B artifact (type: `dataset_manifest`) |
| Per-epoch metrics | `tracker.log_metrics()` — visible as charts in W&B |
| Best checkpoint | W&B artifact (type: `checkpoint`) |
| Train/eval summary | W&B artifact (type: `metrics_summary`) |

### Viewing Results

After a run with `mode: online`, go to your W&B dashboard:
- **Metrics tab**: Loss curves, accuracy over epochs
- **Artifacts tab**: Checkpoints, configs, summaries with full lineage
- **System tab**: GPU/CPU utilization, memory usage

---

## 8. CLI Reference

The CLI has two interfaces: the **Chakra CLI** (`chakra` command) and the **Traditional CLI** (`python -m chakra`).

### Chakra CLI (Recommended)

The Chakra CLI maps each command to a stage of the research cycle:

```bash
chakra <command> --domain DOMAIN [OPTIONS]
```

| Command | Stage | Description |
|---------|-------|-------------|
| `chakra sutra` | Plan | Scaffold version assets and freeze configs |
| `chakra yantra` | Execute | Run training or evaluation (`--stage control\|smoke\|train\|eval`) |
| `chakra rakshak` | Guard | Validate version contract (all required files exist) |
| `chakra vimarsh` | Review | Sync results and generate structured review |
| `chakra manthan` | Improve | Propose ablation suggestions for next iteration |
| `chakra aavart` | Full Cycle | Run the complete loop: Plan → Execute → Guard → Review → Improve |
| `chakra list-domains` | Discovery | List all auto-discovered domains |

**One-command full cycle:**

```bash
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

### Traditional CLI

The traditional entrypoint is `python -m chakra`.

#### Global Options

```bash
python -m chakra [--domain DOMAIN_NAME] COMMAND [OPTIONS]
```

`--domain` is required for all lifecycle commands.

### Discovery Commands

```bash
# List all auto-discovered domains
python -m chakra list-domains

# Show detailed info about a domain
python -m chakra --domain tabular_cls domain-info
```

### Lifecycle Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `scaffold-version` | `--version V` `[--parent P]` `[--lineage scratch\|pretrained]` `[--force]` | Create version assets |
| `validate-version` | `--version V` | Check all files exist |
| `sync-run` | `--version V` `[--source-dir DIR]` `[--wandb-url URL]` `[--dry-run]` | Index outputs into manifest |
| `review-run` | `--version V` | Generate review + roast |
| `next-ablation` | `--version V` | Write ablation suggestions |
| `mirror-obsidian` | `--version V` `[--output-dir DIR]` `[--dry-run]` | Generate Obsidian note |

### Kaggle Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `push-kaggle` | `--version V` `[--title T]` `[--username U]` `[--dry-run]` | Push notebook to Kaggle |
| `kaggle-status` | `--version V` `[--username U]` `[--dry-run]` | Check kernel run status |
| `pull-kaggle` | `--version V` `[--username U]` `[--dry-run]` | Pull outputs into artifacts |

### Execution Orchestration

The `run-execution` command chooses between local and Kaggle execution paths and always runs a local smoke gate before Kaggle submission.

```bash
python -m chakra --domain nlp_lm run-execution --version v1.0 --strategy auto --dry-run
python -m chakra --domain tabular_cls run-execution --version v1.0 --strategy local --dry-run
```

Behavior summary:

- `local` runs the train runner directly.
- `kaggle` performs a local smoke gate first, then push/status/pull.
- `auto` uses manifest lifecycle/execution hints and system info to choose a backend.
- `--dry-run` prints the command flow without invoking the backend tools.

### Runner Commands (Direct)

Runners are invoked directly via Python modules, not through the CLI:

```bash
# Training
python -m chakra.domains.<domain>.train_runner \
  --config <path> --run-name <name> [--device cpu|cuda]

# Evaluation
python -m chakra.domains.<domain>.evaluate_runner \
  --config <path> --run-name <name> --checkpoint <path> [--device cpu|cuda]
```

---

## 9. Working with Each Domain

### Tabular Classification (`tabular_cls`)

**Datasets:** Iris (150 samples, 4 features, 3 classes) and Titanic (891 samples, 9 features, 2 classes)

**Models:**
- `logistic` — Single linear layer (control baseline)
- `mlp` — Two hidden layers with ReLU and dropout

**Metrics:** Accuracy (primary), macro F1, cross-entropy loss

**Iris (v1.0):**
```bash
# Full lifecycle
python -m chakra --domain tabular_cls scaffold-version --version v1.0 --force
python -m chakra.domains.tabular_cls.train_runner --config configs/tabular_cls/v1.0_control.yaml --run-name v1.0-control --device cpu
python -m chakra.domains.tabular_cls.train_runner --config configs/tabular_cls/v1.0_train.yaml --run-name v1.0-train --device cpu
python -m chakra.domains.tabular_cls.evaluate_runner --config configs/tabular_cls/v1.0_train.yaml --run-name v1.0-eval --checkpoint artifacts/v1.0-train/checkpoints/v1.0_train_best.pt --device cpu
python -m chakra --domain tabular_cls sync-run --version v1.0 --source-dir artifacts/v1.0-train
python -m chakra --domain tabular_cls review-run --version v1.0
python -m chakra --domain tabular_cls validate-version --version v1.0
```

**Titanic (v2.0):**
```bash
python -m chakra --domain tabular_cls scaffold-version --version v2.0 --parent v1.0 --force
python -m chakra.domains.tabular_cls.train_runner --config configs/tabular_cls/v2.0_control.yaml --run-name v2.0-control --device cpu
python -m chakra.domains.tabular_cls.train_runner --config configs/tabular_cls/v2.0_train.yaml --run-name v2.0-train --device cpu
python -m chakra.domains.tabular_cls.evaluate_runner --config configs/tabular_cls/v2.0_train.yaml --run-name v2.0-eval --checkpoint artifacts/v2.0-train/checkpoints/v2.0_train_best.pt --device cpu
python -m chakra --domain tabular_cls sync-run --version v2.0 --source-dir artifacts/v2.0-train
python -m chakra --domain tabular_cls review-run --version v2.0
python -m chakra --domain tabular_cls validate-version --version v2.0
```

---

### NLP Language Modelling (`nlp_lm`)

**Dataset:** Tiny Shakespeare (character-level text)

**Models:**
- `bigram` — Bigram baseline (control, no context)
- `gpt_nano` — Minimal GPT transformer (4 layers, 4 heads, 64 embedding dim)

**Metrics:** Bits-per-byte (BPB, primary — lower is better), perplexity, cross-entropy

```bash
# Full lifecycle
python -m chakra --domain nlp_lm scaffold-version --version v1.0 --force
python -m chakra.domains.nlp_lm.train_runner --config configs/nlp_lm/v1.0_control.yaml --run-name v1.0-control --device cpu
python -m chakra.domains.nlp_lm.train_runner --config configs/nlp_lm/v1.0_train.yaml --run-name v1.0-train --device cpu
python -m chakra.domains.nlp_lm.evaluate_runner --config configs/nlp_lm/v1.0_train.yaml --run-name v1.0-eval --checkpoint artifacts/v1.0-train/checkpoints/v1.0_train_best.pt --device cpu
python -m chakra --domain nlp_lm sync-run --version v1.0 --source-dir artifacts/v1.0-train
python -m chakra --domain nlp_lm review-run --version v1.0
python -m chakra --domain nlp_lm validate-version --version v1.0
```

> **Note:** NLP training takes ~5 minutes on CPU. Use `--device cuda` if you have a GPU.

---

### HNDSR Satellite Super-Resolution (`hndsr_vr`)

**Dataset:** Satellite imagery patches (requires local data — see `configs/hndsr_vr/base.yaml`)

**Models:**
- Bicubic baseline (control)
- SR3 diffusion model

**Metrics:** PSNR (primary — higher is better), SSIM

> **Note:** This domain requires the HNDSR dataset configured locally. See `configs/hndsr_vr/local.yaml` for path overrides.

---

## 10. Adding Your Own Domain

AutoResearch is designed to be extended. Adding a new domain requires **zero changes** to the core engine. See [Contributing](contributing.md) for the full tutorial.

### Step-by-Step

**1. Create the domain package:**

```
src/chakra/domains/my_domain/
├── __init__.py
├── domain.yaml
├── lifecycle.py
├── models.py
├── dataset.py
├── metrics.py
├── train_runner.py
└── evaluate_runner.py
```

**2. Write `domain.yaml`:**

```yaml
name: my_domain
display_name: My Research Domain
version_pattern: "^v\\d+\\.\\d+(?:\\.\\d+)?$"
model_kinds: [baseline, main_model]
primary_metric: my_metric
metric_direction: higher_is_better
benchmark_registry: benchmarks/my_domain_registry.json
config_dir: configs/my_domain
programs_doc: programs/my_domain.md
entrypoints:
  lifecycle: chakra.domains.my_domain.lifecycle
  train_runner: chakra.domains.my_domain.train_runner
  evaluate_runner: chakra.domains.my_domain.evaluate_runner
```

**3. Implement `LifecycleHooks` in `lifecycle.py`:**

Use `tabular_cls/lifecycle.py` as your template — it's the simplest complete implementation. You must implement all methods from the `DomainLifecycleHooks` protocol:

| Method | Purpose |
|--------|---------|
| `version_stem()` | Convert `v1.0` → `v1.0_My_Domain` |
| `version_slug()` | Convert `v1.0` → `v1-0-my-domain` |
| `resolve_version_paths()` | Return all canonical file paths for a version |
| `build_version_configs()` | Generate control/smoke/train config dicts |
| `render_notebook()` | Render a Kaggle-ready notebook JSON |
| `render_doc()` | Render version documentation markdown |
| `render_review()` | Render initial review template |
| `render_notebook_readme()` | Render notebooks directory README |
| `default_kernel_metadata()` | Return Kaggle kernel metadata dict |
| `build_findings()` | Analyse a run and return findings + deltas |
| `ablation_suggestions()` | Suggest next-version experiments |
| `roast_lines()` | Return domain-specific roast/audit lines |
| `validate_version()` | Check all required files exist |

**4. Wire W&B tracking in your runners:**

```python
from chakra.core.tracker import init_tracker
from chakra.core.utils import load_dotenv, describe_run_dirs

def main():
    load_dotenv()
    # ... parse args, load config ...
    dirs = describe_run_dirs(config, run_name)
    tracker = init_tracker(config, run_name, dirs["tracker"])

    # During training:
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=epoch)

    # Save artifacts:
    tracker.log_file_artifact("checkpoint", path, "checkpoint")

    # Always finish:
    tracker.finish()
```

**5. Add supporting files:**

```bash
# Config files
configs/my_domain/base.yaml
configs/my_domain/v1.0_control.yaml
configs/my_domain/v1.0_smoke.yaml
configs/my_domain/v1.0_train.yaml

# Benchmark registry (empty initially)
benchmarks/my_domain_registry.json

# Research program doc
programs/my_domain.md
```

**6. Register in `pyproject.toml`:**

```toml
[tool.setuptools.package-data]
"chakra.domains.my_domain" = ["domain.yaml"]
```

**7. Add tests:**

```bash
tests/test_my_domain.py
```

**8. Verify:**

```bash
python -m chakra list-domains   # Should show your domain
python -m pytest tests/test_my_domain.py -v
```

---

## 11. Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Domain Tests

```bash
python -m pytest tests/test_tabular_domain.py -v
python -m pytest tests/test_nlp_domain.py -v
python -m pytest tests/test_core.py -v
```

### Quick Mode

```bash
python -m pytest tests/ -q
```

### Test Coverage Summary

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_core.py` | 16 | Config loading, utils, seeding, domain registry |
| `test_tabular_domain.py` | 9 | Discovery, protocol, models, dataset, metrics |
| `test_nlp_domain.py` | 6 | GPT-nano, bigram, dataset, metrics |
| `test_domain_registry.py` | 5 | Multi-domain discovery, manifest validation |
| `test_cli_dispatch.py` | 3 | CLI argument parsing, domain dispatch |
| `test_runtime_contract.py` | 6 | Path resolution, workspace isolation |
| `test_lifecycle_review.py` | 1 | Full sync → review pipeline |
| `test_notebook_contract.py` | 1 | Notebook JSON structure |

---

## 12. Troubleshooting

### "No module named 'chakra'"

You need to install the package:
```bash
pip install -e ".[dev]"
```

### "ModuleNotFoundError: No module named 'sklearn'"

Install scikit-learn:
```bash
pip install scikit-learn
```

### "No research domains discovered"

Make sure `domain.yaml` files exist in each domain's directory and are listed in `pyproject.toml` under `[tool.setuptools.package-data]`.

### W&B Not Tracking (using NullTracker)

Check:
1. `.env` file exists in the repo root with `WANDB_API_KEY=...`
2. `wandb` is installed: `pip install wandb`
3. Config has `tracking.enabled: true` and `tracking.mode: online`

### "CONFLICT: Merge conflict in README.md"

This happens when merging branches with different histories. Resolve by choosing the version you want:
```bash
git checkout --theirs README.md   # Accept incoming changes
git add README.md
git commit --no-edit
```

### Artifacts Directory Missing

The `artifacts/` directory is gitignored. It's created automatically when you run a training command. If you need to start fresh:
```bash
# Windows
Remove-Item -Recurse -Force artifacts

# Linux/macOS
rm -rf artifacts
```

### GPU Not Detected

Use `--device cpu` to force CPU mode, or check:
```python
import torch
print(torch.cuda.is_available())
```

### Config Inheritance Not Working

Make sure the `inherits:` path is relative to the **repo root**, not the config file:
```yaml
# Correct
inherits: configs/tabular_cls/base.yaml

# Wrong
inherits: base.yaml
```

---

## Summary of All Commands Cheat Sheet

```bash
# ---- Setup ----
pip install -e ".[dev]"
python -m chakra list-domains

# ---- Lifecycle (replace DOMAIN and VERSION) ----
python -m chakra --domain DOMAIN scaffold-version --version VERSION --force
python -m chakra.domains.DOMAIN.train_runner --config configs/DOMAIN/VERSION_control.yaml --run-name VERSION-control --device cpu
python -m chakra.domains.DOMAIN.train_runner --config configs/DOMAIN/VERSION_smoke.yaml --run-name VERSION-smoke --device cpu
python -m chakra.domains.DOMAIN.train_runner --config configs/DOMAIN/VERSION_train.yaml --run-name VERSION-train --device cpu
python -m chakra.domains.DOMAIN.evaluate_runner --config configs/DOMAIN/VERSION_train.yaml --run-name VERSION-eval --checkpoint artifacts/VERSION-train/checkpoints/VERSION_train_best.pt --device cpu
python -m chakra --domain DOMAIN sync-run --version VERSION --source-dir artifacts/VERSION-train
python -m chakra --domain DOMAIN review-run --version VERSION
python -m chakra --domain DOMAIN validate-version --version VERSION

# ---- Chakra CLI (one-command cycle) ----
chakra aavart --domain DOMAIN --version VERSION --device cpu --force

# ---- Testing ----
python -m pytest tests/ -v
```
