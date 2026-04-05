# AutoResearch Quick Start Guide

This guide walks you through a complete end-to-end experiment in under 5 minutes.

---

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **Git** for cloning
- **(Optional)** [W&B account](https://wandb.ai/) for experiment tracking

---

## Step 1: Setup

```bash
# Clone the repo
git clone https://github.com/The-Harsh-Vardhan/autoresearch-by-harsh-vardhan.git
cd autoresearch-by-harsh-vardhan

# Create and activate a virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\activate

# Install everything
pip install -e ".[dev]"
```

### Verify installation

```bash
python -m chakra list-domains
```

You should see three domains: `hndsr_vr`, `nlp_lm`, `tabular_cls`.

---

## Step 2: (Optional) Configure W&B

Create a `.env` file in the repo root:

```
WANDB_API_KEY=your_api_key_here
```

> **Tip:** Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize). If you skip this step, everything still works — metrics are saved locally to JSON.

---

## Step 3: Your First Experiment — Iris Classification

The fastest way to see the full lifecycle is with the tabular classification domain.

### 3a. Scaffold the version

```bash
python -m chakra --domain tabular_cls scaffold-version --version v1.0 --force
```

This creates:
- `notebooks/versions/v1.0_Tabular_CLS.ipynb` — Kaggle-ready notebook
- `docs/notebooks/v1.0_Tabular_CLS.md` — Run documentation
- `reports/reviews/v1.0_Tabular_CLS.review.md` — Review template

### 3b. Run the control baseline

```bash
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v1.0_control.yaml \
  --run-name v1.0-control --device cpu
```

This trains a logistic regression for 1 epoch with 0 train batches — it measures the **untrained baseline** accuracy.

**Expected output:** `Best val accuracy: ~16.67%` (random chance on 3 classes)

### 3c. Train the MLP

```bash
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v1.0_train.yaml \
  --run-name v1.0-train --device cpu
```

**Expected output:** `Best val accuracy: ~93%` after 30 epochs (~10 seconds on CPU)

### 3d. Evaluate

```bash
python -m chakra.domains.tabular_cls.evaluate_runner \
  --config configs/tabular_cls/v1.0_train.yaml \
  --run-name v1.0-eval \
  --checkpoint artifacts/v1.0-train/checkpoints/v1.0_train_best.pt \
  --device cpu
```

**Expected output:** `Accuracy=93.33%, F1=0.9296`

### 3e. Sync, Review, and Validate

```bash
# Index results into a run manifest
python -m chakra --domain tabular_cls sync-run \
  --version v1.0 --source-dir artifacts/v1.0-train

# Generate an automated review and roast
python -m chakra --domain tabular_cls review-run --version v1.0

# Validate the version contract (all files exist)
python -m chakra --domain tabular_cls validate-version --version v1.0
```

**Expected output:** `v1.0 contract passed for domain 'tabular_cls'.`

🎉 **Congratulations!** You've just completed a full lifecycle loop.

---

## Step 4: Try a Different Dataset — Titanic

The same domain supports multiple datasets. Run the Titanic survival prediction:

```bash
# Train
python -m chakra.domains.tabular_cls.train_runner \
  --config configs/tabular_cls/v2.0_train.yaml \
  --run-name v2.0-train --device cpu

# Evaluate
python -m chakra.domains.tabular_cls.evaluate_runner \
  --config configs/tabular_cls/v2.0_train.yaml \
  --run-name v2.0-eval \
  --checkpoint artifacts/v2.0-train/checkpoints/v2.0_train_best.pt \
  --device cpu
```

**Expected output:** `Accuracy=83.63%, F1=0.8116`

---

## Step 5: Run the NLP Domain

```bash
# Scaffold
python -m chakra --domain nlp_lm scaffold-version --version v1.0 --force

# Control baseline (bigram model)
python -m chakra.domains.nlp_lm.train_runner \
  --config configs/nlp_lm/v1.0_control.yaml --run-name v1.0-control --device cpu

# Full training (GPT-nano, 5 epochs)
python -m chakra.domains.nlp_lm.train_runner \
  --config configs/nlp_lm/v1.0_train.yaml --run-name v1.0-train --device cpu

# Evaluate
python -m chakra.domains.nlp_lm.evaluate_runner \
  --config configs/nlp_lm/v1.0_train.yaml --run-name v1.0-eval \
  --checkpoint artifacts/v1.0-train/checkpoints/v1.0_train_best.pt --device cpu
```

**Expected results:**
- Bigram baseline: BPB = 6.38, Perplexity = 83.28
- GPT-nano trained: BPB = 3.59, Perplexity = 12.18 (77% improvement)

---

## Step 6: Run the Tests

```bash
python -m pytest tests/ -v
```

All 47 tests should pass.

---

## Understanding the Config System

Configs use YAML with an `inherits:` key for layered configuration:

```yaml
# configs/tabular_cls/v1.0_train.yaml
inherits: configs/tabular_cls/base.yaml    # ← inherit from base

project:
  group: v1.0-train                         # ← override specific fields

training:
  epochs: 30
  checkpoint_name: v1.0_train_best.pt
```

**Config variants per version:**
- `*_control.yaml` — Baseline model, no training (sets the floor)
- `*_smoke.yaml` — Limited batches, fast pipeline validation
- `*_train.yaml` — Full training run

---

## Understanding the Domain Protocol

Every domain implements `DomainLifecycleHooks`:

```python
class LifecycleHooks:
    def version_stem(self, version: str) -> str: ...
    def resolve_version_paths(self, version: str) -> VersionPaths: ...
    def build_version_configs(self, version, parent, lineage) -> dict: ...
    def render_notebook(self, version, parent, paths) -> str: ...
    def render_doc(self, version, parent, paths) -> str: ...
    def render_review(self, version) -> str: ...
    def build_findings(self, version, manifest, eval_summary, registry): ...
    def ablation_suggestions(self, version, eval_summary, delta): ...
    def roast_lines(self) -> list[str]: ...
    def validate_version(self, version) -> list[str]: ...
```

The core `lifecycle.py` calls these hooks — it never contains domain-specific logic.

---

## Next Steps

- **Add W&B tracking** — Set up your `.env` and rerun experiments to see them in the W&B dashboard
- **Add a new domain** — Follow the guide in [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Push to Kaggle** — Use `push-kaggle` and `pull-kaggle` commands for cloud execution
- **Extend the framework** — Add custom metrics, models, or datasets to existing domains
