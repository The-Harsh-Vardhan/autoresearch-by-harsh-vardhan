# Concept vs Reality Audit

**Date:** 2026-04-02  
**Auditor:** Antigravity  
**Scope:** Every module, every claim, every contract

---

## Grading Scale

| Grade | Meaning |
|-------|---------|
| ✅ A | Concept matches reality. Works end-to-end. |
| ⚠️ B | Structure correct, has gaps or dead code. |
| 🟡 C | Exists but doesn't actually work or is incomplete. |
| 🔴 D | Concept claims it exists, reality says otherwise. |

---

## 1. Core Engine (`core/`)

### 1.1 `interfaces.py` — DomainLifecycleHooks Protocol

**Concept:** A clean Protocol that every domain must implement. 14 methods.

**Reality:** ✅ **A** — The protocol is well-defined, both domains implement all 14 methods, and `@runtime_checkable` allows isinstance checks. `VersionPaths` dataclass is clean and frozen.

**But:** The protocol uses `...` (Ellipsis) as the body for every method. This is correct Python Protocol syntax but provides zero docstring guidance for implementors beyond the method signature.

---

### 1.2 `domain_registry.py` — Auto-Discovery

**Concept:** Scans `domains/*/domain.yaml` to discover all installed domains. Returns structured `DomainManifest` objects.

**Reality:** ✅ **A** — Works. Discovers both domains. The YAML parser handles all syntax used in domain.yaml files. `load_lifecycle_hooks()` dynamically imports the lifecycle module.

---

### 1.3 `lifecycle.py` — Generic Orchestration

**Concept:** 9 lifecycle commands (scaffold, validate, push, status, pull, sync, review, mirror, ablation) that work for any domain.

**Reality:** ⚠️ **B** — All 9 functions exist and have correct signatures. But:

| Issue | Severity |
|-------|----------|
| `_next_version_labels()` has HNDSR-specific logic (`vR.P.` prefix handling) hardcoded in a "generic" function | MEDIUM |
| `_write_yaml()` falls back to JSON when pyyaml is missing — scaffolded configs will be `.yaml` files containing JSON | LOW |
| `review_run()` line 219 is a single 450+ character format string — unreadable, unmaintainable | LOW |
| `mirror_obsidian()` line 247 is another 300+ character format string | LOW |
| No error handling if `build_findings()` or `ablation_suggestions()` raise | LOW |

---

### 1.4 `tracker.py` — Experiment Tracking

**Concept:** W&B tracker with graceful fallback to a local NullTracker.

**Reality:** ✅ **A** — Clean inheritance. NullTracker writes local JSON records. WandbTracker wraps the W&B API with local fallback. `init_tracker()` handles every failure mode (import failure, init failure, disabled mode). No torch dependency.

---

### 1.5 `utils.py` — Shared Utilities

**Concept:** Domain-agnostic config loading, seeding, path helpers. No ML deps at import time.

**Reality:** ⚠️ **B** — Mostly correct, but:

| Issue | Severity |
|-------|----------|
| `get_device()` imports torch at call time but **doesn't** have a try/except — will crash without torch even though it's in "core" | MEDIUM |
| `REPO_ROOT = Path(__file__).resolve().parents[3]` assumes a fixed 3-level nesting — fragile if the package is installed elsewhere | LOW |
| `_parse_simple_yaml()` works for flat and one-level-nested YAML but will break on deeply nested configs (e.g., nested `paths.datasets.kaggle_4x.hr_dir`) | LOW |

---

## 2. HNDSR Domain (`domains/hndsr_vr/`)

### 2.1 `lifecycle.py` — HNDSR LifecycleHooks

**Concept:** Full DomainLifecycleHooks implementation for satellite super-res.

**Reality:** ✅ **A** — Massive (400+ lines), implements all 14 protocol methods. Config building, notebook rendering, validation, findings, roast — all present. Has been the existing production path since before the refactor.

---

### 2.2 `models.py`, `dataset.py`, `metrics.py`

**Concept:** SR3 diffusion model, satellite dataset with synthetic/Kaggle lanes, PSNR/SSIM metrics.

**Reality:** ⚠️ **B** — Cannot verify at runtime (torch missing), but the code is structurally present and was ported from the pre-refactor working version. The test suite validates these when torch is available.

---

### 2.3 `train_runner.py`, `evaluate_runner.py`

**Concept:** Standalone CLI runners that load config, build model/data, train/evaluate, save artifacts.

**Reality:** ⚠️ **B** — Code present, structurally correct. Need torch to verify.

---

## 3. NLP Domain (`domains/nlp_lm/`)

### 3.1 `lifecycle.py` — NLP LifecycleHooks

**Concept:** Full lifecycle hooks for character-level LM research.

**Reality:** ⚠️ **B** — Implements all 14 methods, but:

| Issue | Severity |
|-------|----------|
| `render_notebook()` generates a near-empty notebook — no training cells, no evaluation cells, no subprocess calls to train_runner | HIGH |
| HNDSR notebook has full subprocess calls (`python -m autoresearch_hv.hndsr_vr.train_runner --config ...`), NLP has none | HIGH |
| `ablation_suggestions()` returns hardcoded strings regardless of actual eval results | MEDIUM |
| `render_doc()` is a 3-line template vs HNDSR's detailed documentation | LOW |

---

### 3.2 `models.py` — GPTNano + BigramBaseline

**Concept:** A small GPT transformer and a bigram baseline.

**Reality:** 🟡 **C** — Code exists but:

| Issue | Severity |
|-------|----------|
| GPTNano has no `training_step()` method unlike HNDSR's SR3 which returns `(loss, stats)` | HIGH |
| The train_runner must compute loss externally — inconsistent API across domains | MEDIUM |
| BigramBaseline is correct but trivially simple (lookup table) | LOW |

---

### 3.3 `dataset.py` — CharTextDataset

**Concept:** Character-level text dataset with tokenizer.

**Reality:** ⚠️ **B** — Clean implementation with `build_loaders()`. But:

| Issue | Severity |
|-------|----------|
| `build_loaders()` hardcodes `data/tiny_shakespeare.txt` path lookup through `config["paths"]["datasets"]` — no fallback | LOW |
| The dataset is tiny (6KB) — good for smoke tests, too small for real experiments | LOW |

---

### 3.4 `train_runner.py` — NLP Training

**Concept:** Standalone training entry point.

**Reality:** 🟡 **C** — Code exists but:

| Issue | Severity |
|-------|----------|
| Has `if __name__ == "__main__"` block but the NLP notebook never calls it | HIGH |
| No `--config` argument parsing — uses `argparse` but config path is hardcoded to `configs/nlp_lm/base.yaml` | HIGH |
| Training loop is present but has never been executed | MEDIUM |

---

### 3.5 `evaluate_runner.py` — NLP Evaluation

**Concept:** Standalone evaluation entry point.

**Reality:** 🟡 **C** — Same issues as train_runner. Never been executed. Config path handling is inconsistent.

---

## 4. Backwards-Compat Shims (`hndsr_vr/`)

**Concept:** Thin re-exports so old `autoresearch_hv.hndsr_vr.*` imports still work.

**Reality:** ⚠️ **B** — Works for verified paths, but:

| Issue | Severity |
|-------|----------|
| `hndsr_vr/__init__.py` uses `from autoresearch_hv.domains.hndsr_vr import *` — wildcard import pulls in `LifecycleHooks` which triggers torch import chain | MEDIUM |
| Other shims (`dataset.py`, `models.py`, etc.) are simple re-exports that work | LOW |

---

## 5. CLI (`cli.py`)

**Concept:** Unified entrypoint with `--domain` dispatch.

**Reality:** ✅ **A** — Clean, working. Verified end-to-end. Proper domain guard.

**But:** 
| Issue | Severity |
|-------|----------|
| No `--version` flag for `domain-info` — should show the version's available configs | LOW |
| No `--help` examples in subcommand descriptions | LOW |

---

## 6. Infrastructure

### 6.1 Configs

| Domain | Concept | Reality | Grade |
|--------|---------|---------|-------|
| HNDSR | 4 configs (base, control, smoke, train) | ✅ All exist, properly structured | A |
| NLP | 4 configs | ✅ All exist now | A |

### 6.2 Tests

| File | Tests | Pass? | Grade |
|------|-------|-------|-------|
| `test_domain_registry.py` | 5 | ✅ All pass | A |
| `test_cli_dispatch.py` | 3 | ✅ All pass | A |
| `test_nlp_domain.py` | 6 | ⏭️ Skip (needs torch) | B |
| `test_runtime_contract.py` | 6 | ⏭️ Skip (needs torch) | B |
| `test_lifecycle_review.py` | 1 | ⏭️ Skip (needs torch) | B |
| `test_notebook_contract.py` | 1 | ⏭️ Skip (needs torch) | B |

**Missing tests:**
- No test for `core/lifecycle.py` functions (scaffold, validate, etc.) — these could run without torch
- No test for `core/tracker.py` NullTracker serialization — could run without torch
- No test for the YAML fallback parser edge cases
- No test for `core/utils.py` config loading with `inherits`

---

## 7. Overall Verdict

| Component | Grade | Status |
|-----------|-------|--------|
| Core engine | **A-** | Solid, one fragile function (`get_device`) |
| HNDSR domain | **A-** | Mature, ported from working code |
| NLP domain | **C+** | Scaffolded, notebook and runners never tested |
| Backwards shims | **B+** | Work, minor wildcard import concern |
| CLI | **A** | Clean, verified |
| Infrastructure | **B+** | Configs/data complete, tests incomplete |

**The weakest link is the NLP domain.** It's a proof-of-concept that was scaffolded but never run end-to-end. The notebook is empty. The train/evaluate runners have config path issues. The models lack a consistent `training_step()` API.

---

## 8. Priority Fixes

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| 1 | Fix `get_device()` to handle missing torch | Core reliability | 2 min |
| 2 | NLP `train_runner.py` — fix config argparse | NLP usability | 10 min |
| 3 | NLP `evaluate_runner.py` — fix config argparse | NLP usability | 10 min |
| 4 | NLP notebook — add actual training/eval subprocess cells | NLP completeness | 15 min |
| 5 | Break up monster format strings in `core/lifecycle.py` | Readability | 10 min |
| 6 | Remove HNDSR-specific logic from `_next_version_labels()` | Core purity | 5 min |
| 7 | Add core-only tests (lifecycle scaffold, tracker, YAML parser, config inherits) | Test coverage | 20 min |
| 8 | Fix wildcard import in `hndsr_vr/__init__.py` | Import safety | 2 min |
