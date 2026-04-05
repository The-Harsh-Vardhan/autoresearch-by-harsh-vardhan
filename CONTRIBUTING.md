# Contributing to AutoResearch

## Development Setup

```bash
# Clone the repo
git clone https://github.com/The-Harsh-Vardhan/autoresearch-by-harsh-vardhan.git
cd autoresearch-by-harsh-vardhan

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode (includes pytest)
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full suite (47 tests)
python -m pytest tests/ -v

# Quick mode
python -m pytest tests/ -q

# Run a specific domain's tests
python -m pytest tests/test_tabular_domain.py -v
```

Tests that require `torch` are **skipped automatically** if PyTorch is not installed. Core infrastructure tests always run.

## Project Structure

See [README.md](README.md) for the full layout. The key principle is:

- **`core/`** — Domain-agnostic engine. Never import domain-specific code here.
- **`domains/*/`** — Self-contained research plugins. May import from `core/` only.
- **`hndsr_vr/`** (root-level) — Backwards-compat shims. Thin re-exports only, no logic.

## Adding a New Domain

Adding a new research domain requires **zero changes** to the core engine:

### 1. Create the domain package

```
src/autoresearch_hv/domains/your_domain/
├── __init__.py
├── domain.yaml          # Auto-discovery manifest
├── lifecycle.py         # LifecycleHooks class
├── models.py            # Your model architectures
├── dataset.py           # Data loading and preprocessing
├── metrics.py           # Domain-specific metrics
├── train_runner.py      # Training entrypoint
└── evaluate_runner.py   # Evaluation entrypoint
```

### 2. Write the `domain.yaml` manifest

```yaml
name: your_domain
display_name: Your Domain Name
version_pattern: "^v\\d+\\.\\d+(?:\\.\\d+)?$"
model_kinds:
  - baseline_model
  - main_model
primary_metric: your_metric
metric_direction: higher_is_better  # or lower_is_better
benchmark_registry: benchmarks/your_domain_registry.json
config_dir: configs/your_domain
programs_doc: programs/your_domain.md
entrypoints:
  lifecycle: autoresearch_hv.domains.your_domain.lifecycle
  train_runner: autoresearch_hv.domains.your_domain.train_runner
  evaluate_runner: autoresearch_hv.domains.your_domain.evaluate_runner
```

### 3. Implement `LifecycleHooks` in `lifecycle.py`

Your hooks class must implement the `DomainLifecycleHooks` protocol (see `core/interfaces.py`). Use `tabular_cls/lifecycle.py` as a reference — it's the simplest complete implementation.

### 4. Wire W&B tracking in runners

```python
from autoresearch_hv.core.tracker import init_tracker
from autoresearch_hv.core.utils import load_dotenv

def main():
    load_dotenv()                          # Load .env credentials
    tracker = init_tracker(config, run_name, dirs["tracker"])
    # ... training loop ...
    tracker.log_metrics(metrics, step=epoch)
    tracker.log_file_artifact(name, path, type)
    tracker.finish()
```

### 5. Add supporting files

- `configs/your_domain/base.yaml` + version configs
- `benchmarks/your_domain_registry.json` (empty, populated after first control run)
- `programs/your_domain.md` (research program doc)
- One line in `pyproject.toml` → `[tool.setuptools.package-data]`

### 6. Add tests

- `tests/test_your_domain.py` — discovery, protocol compliance, model shapes, metrics

### 7. Verify

```bash
# Should show your domain
python -m autoresearch_hv list-domains

# All tests should pass
python -m pytest tests/ -v
```

## Code Standards

- Follow bounded coding discipline (inspired by [Power of 10](https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code))
- Keep control flow simple and bounded
- Every function that can fail must check its return value
- Validate inputs at module boundaries
- Use type hints everywhere

## Commit Convention

Use conventional commits:

| Prefix | Scope | Example |
|--------|-------|---------|
| `feat` | `domain`, `core` | `feat(domain): add tabular_cls domain with Iris and Titanic` |
| `fix` | `core`, `cli` | `fix(core): resolve config inheritance for nested keys` |
| `test` | — | `test: add 9 tests for tabular_cls domain` |
| `docs` | — | `docs: add quickstart guide and showcase README` |
| `refactor` | — | `refactor: extract deep_merge into separate utility` |

## Checklist for PRs

- [ ] All tests pass (`python -m pytest tests/ -q`)
- [ ] New domain auto-discovered (`python -m autoresearch_hv list-domains`)
- [ ] Version contract validates (`validate-version`)
- [ ] Configs use `inherits:` pattern from `base.yaml`
- [ ] W&B tracking wired via `init_tracker()` + `load_dotenv()`
- [ ] Benchmark registry has real measured baselines
