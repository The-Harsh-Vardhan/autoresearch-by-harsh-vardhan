# HNDSR vR Research Program

This file is the human-owned research program for the HNDSR scratch lineage.

## Objective

Improve the `vR.x` HNDSR scratch lane through bounded, reviewable experiments that survive GitHub, Kaggle, W&B, and post-run audit without notebook-only logic.

## Fixed Contracts

- Dataset manifests and split semantics are fixed per version.
- Evaluation summaries, qualitative sample grids, and benchmark comparisons are fixed outputs.
- Every frozen version must have a notebook, an external doc, and a review/roast doc.
- W&B online logging is required for milestone completion.

## Allowed Mutation Surface

- New version scaffolds under `notebooks/versions/`, `docs/notebooks/`, `reports/reviews/`, and `configs/hndsr_vr/`
- HNDSR runtime modules under `src/chakra/hndsr_vr/`
- Benchmark registry and review heuristics

## Disallowed Mutation Surface

- Ad hoc notebook-only training logic
- Untracked Kaggle-only patches that are not mirrored back into repo code or docs
- Version promotion without a written review and next-step recommendation

## Version Rules

- `vR.N` is a major step: model family, optimizer/loss family, dataset/protocol, evaluation contract, or branch promotion changed.
- `vR.N.M` is a minor bounded change within the same lane.
- Runtime-only Kaggle fixes stay in the same version until the review closes.

## Promotion Rules

- Freeze only after Kaggle outputs are pulled, synced, reviewed, and mirrored.
- Fork the next version only after the parent review records a promotion decision and 1-3 bounded ablations.
- Prefer single-variable ablations whenever practical.

## Git Discipline

- Commit before Kaggle handoff.
- Commit after Kaggle runtime fixes are synced.
- Commit after the review/roast is frozen.
- Use version-bound commit prefixes such as `research(vR.1):`, `runtime(vR.1):`, and `review(vR.1):`.

## Engineering Baseline

Apply Power-of-10 discipline conservatively:

- Keep control flow simple.
- Bound loops, retries, and polling.
- Keep runtime state scoped and explicit.
- Validate inputs and check fallible returns.
- Favor zero-warning, analyzable code paths.
