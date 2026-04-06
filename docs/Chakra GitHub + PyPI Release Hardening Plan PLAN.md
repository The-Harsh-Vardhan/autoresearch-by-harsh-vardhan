# Chakra GitHub + PyPI Release Hardening Plan

## Summary

- Convert Chakra from a repo-shaped project into an install-shaped package. The current runtime assumes a checkout with `configs/`, `benchmarks/`, `data/`, `reports/`, and `notebooks/` beside the code, which is the main reason it is not PyPI-ready.
- Standardize the public release surface around `chakra-auto-research` on PyPI, `chakra` as the import package, and `chakra` / `python -m chakra` as consistent CLI entrypoints.
- Ship a lighter default install and move heavyweight ML/tracking dependencies into extras, while making missing-extra failures explicit and helpful.
- Add GitHub release/security automation: CI build validation, wheel install smoke tests, dependency/security scans, and Trusted Publisher workflows for TestPyPI/PyPI.
- Fix the README/PyPI presentation so the logo is actually visible on PyPI and the first-run instructions work from a clean `pip install`.

## Key Changes

### 1. Packaging and runtime model

- Replace hardcoded `REPO_ROOT` runtime assumptions with two explicit concepts:
  - packaged read-only resources loaded via `importlib.resources`
  - user workspace root for mutable outputs
- Bundle all runtime-required immutable assets inside the wheel/sdist:
  - domain manifests
  - default configs
  - benchmark registries
  - packaged sample data needed for supported demos (`titanic.csv`, `tiny_shakespeare.txt`)
- Keep generated outputs out of the package and write them into a workspace rooted at:
  - `--workspace PATH` when provided
  - `CHAKRA_WORKSPACE` when set
  - current working directory by default
- Make every path-producing command write under the workspace instead of assuming the repository exists locally.
- Keep HNDSR installable but optional; it remains data-dependent and should fail with a clear dataset/setup message rather than a path crash.

### 2. Public interfaces and dependency layout

- Unify CLI behavior so `chakra` and `python -m chakra` use the same parser and commands.
- Keep existing legacy command paths and `chakra.hndsr_vr.*` compatibility shims for one release; document them as compatibility surfaces, not the preferred API.
- Remove the package-root CLI import side effect from `import chakra`; expose `chakra.__version__` from installed metadata instead of a hardcoded string.
- Split dependencies into explicit extras:
  - base: only lightweight package/runtime dependencies
  - `tabular`: tabular execution dependencies
  - `nlp`: NLP execution dependencies
  - `hndsr`: HNDSR execution dependencies
  - `tracking`: W&B integration
  - `dev`: test/build/audit tooling
  - `all`: union of runtime extras
- Add dependency guards so domain commands fail with install guidance like `pip install "chakra-auto-research[tabular]"` instead of import tracebacks.
- Standardize docs/badges/install strings on the normalized PyPI name `chakra-auto-research`.

### 3. GitHub readiness, release automation, and security hardening

- Update package metadata for modern packaging:
  - SPDX license string
  - license files
  - consistent version source
  - clean project URLs
  - accurate classifiers/extras
- Add CI that validates:
  - tests
  - package build
  - `twine check`
  - clean-venv install smoke from built wheel
  - CLI smoke after install
- Add security controls:
  - `SECURITY.md`
  - Dependabot for Python and GitHub Actions
  - CodeQL workflow
  - dependency-review on PRs
  - `pip-audit` gate for shipped dependencies
  - minimal GitHub workflow permissions
- Add Trusted Publisher release workflows:
  - TestPyPI publish on manual dispatch and/or prerelease tags
  - PyPI publish on release/tag
  - no long-lived PyPI API token stored in GitHub
- Treat “no vulnerabilities” as:
  - no obvious first-party security footguns in path/subprocess handling
  - no committed secrets
  - zero unresolved known dependency vulnerabilities at release cut time

### 4. README, PyPI page, and ease-of-use

- Replace the current white logo in the README header with the visible non-white logo asset for PyPI rendering.
- Keep the image as an absolute HTTPS URL hosted from GitHub, since PyPI long descriptions cannot rely on packaged local image paths.
- Rewrite the top of the README around supported install paths:
  - `pip install chakra-auto-research`
  - `pip install "chakra-auto-research[tabular]"`
  - `pip install "chakra-auto-research[all]"`
- Make the first supported success path a clean PyPI install plus a CPU-friendly tabular quickstart.
- Add explicit “works after pip install” guarantees:
  - `chakra list-domains`
  - `chakra domain-info --name tabular_cls`
  - tabular demo with the `tabular` extra
- Move repo-clone/editable-install instructions below the PyPI flow instead of presenting them as the primary onboarding path.

## Test Plan

- Packaging validation:
  - `python -m build`
  - `python -m twine check dist/*`
  - fresh venv install from the built wheel
- Install-smoke validation:
  - base install: import `chakra`, print version, `chakra list-domains`
  - tabular extra: run a workspace-scoped tabular lifecycle smoke path on CPU
  - missing-extra path: verify friendly install hint instead of traceback
- Runtime-path validation:
  - run from outside the repo
  - run with `--workspace`
  - run with `CHAKRA_WORKSPACE`
- Test-suite hardening:
  - convert tests to `tmp_path`/workspace-driven outputs instead of writing into checked-in `artifacts/`
  - add installed-wheel tests so CI covers non-editable installs, not only source checkout behavior
- Security validation:
  - `pip-audit` clean on release branch
  - dependency review on PRs
  - workflow permissions and Trusted Publisher flow verified on TestPyPI before first real publish
- README/PyPI validation:
  - confirm rendered long description on TestPyPI
  - confirm the logo is visible on a white background

## Assumptions and defaults

- Default install strategy is `core + extras`.
- Default publishing strategy is GitHub Trusted Publishing, with TestPyPI used before first real PyPI release.
- Canonical public package name is `chakra-auto-research`; import name remains `chakra`.
- HNDSR remains an optional lane that requires user-supplied dataset configuration.
- Backward compatibility is preserved for the current legacy CLI/module surfaces for one release cycle.
- The goal is “no known vulnerabilities at release time,” not a blanket guarantee against future third-party CVEs after publication.
