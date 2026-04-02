# Contributing to AutoResearch

## Development Setup

```bash
# Clone the repo
git clone https://github.com/The-Harsh-Vardhan/autoresearch-by-harsh-vardhan.git
cd autoresearch-by-harsh-vardhan

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Set the PYTHONPATH
export PYTHONPATH=src  # or $env:PYTHONPATH = "src" on PowerShell

# Run the test suite
python -m pytest tests/ -v
```

Tests that require `torch` will be **skipped automatically** if PyTorch is not installed. The core infrastructure tests (domain registry, CLI dispatch) always run.

## Project Structure

See [README.md](README.md) for the full layout. The key principle is:

- **`core/`** — Domain-agnostic engine. Never import domain-specific code here.
- **`domains/*/`** — Self-contained research plugins. May import from `core/` only.
- **`hndsr_vr/`** — Backwards-compat shims. Thin re-exports only, no logic.

## Adding a New Domain

1. Create `src/autoresearch_hv/domains/<your_domain>/`
2. Add a `domain.yaml` manifest (copy from an existing domain)
3. Implement the `DomainLifecycleHooks` protocol in `lifecycle.py`
4. Add domain-specific `models.py`, `dataset.py`, `metrics.py`, and runners
5. Create configs under `configs/<your_domain>/`
6. Create a benchmark registry under `benchmarks/<your_domain>_registry.json`
7. Create a program doc under `programs/<your_domain>.md`
8. Add tests under `tests/test_<your_domain>.py`
9. The CLI auto-discovers your domain — no registration code needed

## Code Standards

- Follow [Power of 10](programs/hndsr_vr.md) coding discipline
- Keep control flow simple and bounded
- Every function that can fail must check its return value
- No dynamic allocation after init where possible
- Validate inputs at module boundaries

## Commit Convention

Use conventional commits:
- `feat(domain):` New domain or feature
- `fix(core):` Bug fix in the core engine
- `test:` Test additions or fixes
- `docs:` Documentation changes
- `refactor:` Code restructuring without behavior change
