# The Chakra Cycle (Aavart)

This document explains each stage of the Chakra research cycle in detail.

---

## Overview

Chakra is built around one idea: **research is cyclical**. Every experiment follows the same loop:

```
Plan → Execute → Guard → Review → Improve → Repeat
```

In Chakra's terminology, a complete loop is called an **Aavart** (आवर्त — "revolution" or "cycle"). Each stage within the Aavart has a Sanskrit name that reflects its role:

| # | Stage | Chakra Name | Sanskrit | Meaning |
|---|-------|-------------|----------|---------|
| 1 | Plan | **Sutra** | सूत्र | Thread / Formula — the plan that holds the experiment together |
| 2 | Execute | **Yantra** | यन्त्र | Instrument / Machine — the engine that produces results |
| 3 | Guard | **Rakshak** | रक्षक | Guardian — protects integrity by validating contracts |
| 4 | Review | **Vimarsh** | विमर्श | Reflection / Analysis — examines what happened and why |
| 5 | Improve | **Manthan** | मन्थन | Churning — extracts insight from results, proposes what's next |

---

## Stage 1: Sutra (Plan)

**English:** Create and freeze the experiment configuration.

**What happens:**
- The system scaffolds all version assets: notebook, documentation, review template, config files
- Three config variants are generated: `control`, `smoke`, `train`
- Configs are **frozen** after creation — this prevents goal-drift during execution

**Why it matters:**
Research without a frozen plan is just tinkering. The Sutra ensures that every experiment starts with a clear, immutable hypothesis. If you want to change something, you create a new version — you never mutate the current plan.

**Command:**
```bash
chakra sutra --domain tabular_cls --version v1.0 --force
```

**What gets created:**
```
notebooks/versions/v1.0_Tabular_CLS.ipynb    # Kaggle-ready notebook
docs/notebooks/v1.0_Tabular_CLS.md           # Run documentation
reports/reviews/v1.0_Tabular_CLS.review.md   # Review template
configs/tabular_cls/v1.0_control.yaml         # Baseline config
configs/tabular_cls/v1.0_smoke.yaml           # Quick-check config
configs/tabular_cls/v1.0_train.yaml           # Full training config
```

---

## Stage 2: Yantra (Execute)

**English:** Run the training and evaluation pipeline.

**What happens:**
Execution occurs in four sub-stages, each with a specific purpose:

| Sub-stage | Config Used | Purpose |
|-----------|------------|---------|
| **Control** | `*_control.yaml` | Run the baseline model with minimal/no training. This sets the accuracy floor. |
| **Smoke** | `*_smoke.yaml` | Run the main model with heavily limited batches. Validates the pipeline works end-to-end. |
| **Train** | `*_train.yaml` | Full training run. Produces the real results. |
| **Eval** | `*_train.yaml` + checkpoint | Load the best checkpoint and evaluate on the validation set. |

**Why it matters:**
The control → smoke → train → eval progression catches failures early. If the control baseline doesn't run, there's no point starting full training. If smoke fails, there's a bug in the pipeline. This saves hours of wasted GPU time.

**Commands:**
```bash
# Run all four sub-stages:
chakra yantra --domain tabular_cls --version v1.0 --stage control --device cpu
chakra yantra --domain tabular_cls --version v1.0 --stage smoke --device cpu
chakra yantra --domain tabular_cls --version v1.0 --stage train --device cpu
chakra yantra --domain tabular_cls --version v1.0 --stage eval --device cpu
```

**Or let Aavart run them all automatically:**
```bash
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

---

## Stage 3: Rakshak (Guard)

**English:** Validate that all contracts are satisfied.

**What happens:**
- Checks that all required files exist (notebook, doc, review, configs)
- Verifies the version structure matches the domain's contract
- Reports any missing or malformed assets

**Why it matters:**
Without validation, you can accidentally publish incomplete experiments. The Rakshak ensures that nothing leaves the system without meeting the minimum quality bar.

**Command:**
```bash
chakra rakshak --domain tabular_cls --version v1.0
```

**Expected output:**
```
  🛡️ Rakshak (Guard): Validating version contract...
  🛡️ Rakshak (Guard): ✓ Contract passed
v1.0 contract passed for domain 'tabular_cls'.
```

---

## Stage 4: Vimarsh (Review)

**English:** Sync results and generate a structured review.

**What happens:**
1. **Sync** — Indexes training outputs (checkpoints, eval summaries, metrics) into a structured `run_manifest.json`
2. **Review** — Compares results against the benchmark registry, computes metric deltas, generates severity-ordered findings, and produces a "roast" (candid assessment)

**Why it matters:**
Raw metrics are useless without interpretation. The Vimarsh turns numbers into actionable findings: "accuracy improved by 76.7pp over baseline" or "checkpoint is missing — critical failure."

**Command:**
```bash
chakra vimarsh --domain tabular_cls --version v1.0
```

**What gets written:**
```
reports/reviews/v1.0_Tabular_CLS.review.md      # Human-readable review
artifacts/runs/v1.0/review_payload.json           # Machine-readable findings
```

**Example review snippet:**
```markdown
## Findings
- [INFO] eval_summary found: accuracy=0.9333, f1=0.9296

## Roast
- Where is your ablation plan?
- Single seed results are noise, not science.

## Promotion Decision
- Decision: freeze and fork next version
```

---

## Stage 5: Manthan (Improve)

**English:** Propose bounded improvements for the next iteration.

**What happens:**
- Analyzes the review findings and metric deltas
- Generates concrete, bounded ablation suggestions
- Writes suggestions to `reports/generated/v1.0_next_ablation.md`

**Why it matters:**
Manthan (मन्थन) means "churning" — it references the mythological churning of the ocean that produced both poison and nectar. Similarly, this stage extracts valuable insights from potentially noisy results, and proposes specific, testable improvements rather than vague "try more things."

**Command:**
```bash
chakra manthan --domain tabular_cls --version v1.0
```

**Example output:**
```
# Next Ablations

- Try hidden_dim=128 (currently 64) — network may be capacity-limited
- Add learning rate scheduler (cosine annealing)
- Run with 3 seeds to establish confidence intervals
```

---

## The Full Cycle: Aavart

**English:** Run all five stages as a single orchestrated loop.

The `aavart` command chains everything together:

```bash
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

**What Aavart does internally:**

```
1. Sutra    → scaffold_version()
2. Yantra   → train_runner (control) → train_runner (smoke) → train_runner (train) → evaluate_runner
3. Vimarsh  → sync_run() → review_run()
4. Rakshak  → validate_version()
5. Manthan  → next_ablation()
```

**After Aavart completes**, you have:
- A frozen plan with all configs
- Control baseline, smoke test, and full training results
- An evaluated best checkpoint
- A structured review with metric deltas
- Ablation suggestions for v1.1

**Then the cycle repeats:**
```bash
chakra aavart --domain tabular_cls --version v1.1 --device cpu --force
```

---

## Chakra Logging

Throughout every stage, Chakra emits structured logs so you always know where you are in the cycle:

```
🔁 [Chakra] Starting Aavart (Full Cycle) — tabular_cls v1.0
   Plan → Execute → Guard → Review → Improve

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

Each line follows the format:
```
  {emoji} {Chakra Name} ({English}): {message}
```

If a stage fails:
```
❌ [Chakra] Aavart failed at Yantra (Execute) — tabular_cls v1.0
   Error: Training failed for v1.0-train (exit code 1)
```

---

## Design Principles

1. **English provides clarity, Chakra provides identity.** Every term has both.
2. **Plans are frozen.** Once a Sutra is woven, it doesn't change. Fork a new version instead.
3. **Guard before you celebrate.** Rakshak runs after Review, not before — catching issues in the output, not just the input.
4. **Churning produces both poison and nectar.** Manthan's suggestions are bounded and testable, not open-ended wishlists.
5. **The cycle never stops.** Each Aavart ends with the seeds for the next one.
