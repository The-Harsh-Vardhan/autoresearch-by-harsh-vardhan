# AutoResearch by Harsh Vardhan

`AutoResearch by Harsh Vardhan` is a standalone research platform for the HNDSR scratch lineage. It keeps the core spirit of `karpathy/autoresearch` while replacing the single-file LLM benchmark loop with a stricter notebook lifecycle for Kaggle, GitHub, Weights & Biases, and post-run review.

## Phase 1 Contract

- First operational lane: `vR.x` HNDSR scratch research
- Phase-one success: `vR.1` completes GitHub source control, Kaggle execution, W&B online logging, pullback, review/roast, and Obsidian mirror
- Notebooks stay thin and immutable once frozen
- Repo modules own train, eval, export, version validation, and post-run analysis

## Layout

```text
programs/                  Human-owned research org instructions
src/autoresearch_hv/       CLI and HNDSR runtime modules
configs/hndsr_vr/          Base + versioned configs
notebooks/versions/        Immutable Kaggle notebook versions
docs/notebooks/            External per-version run docs
reports/reviews/           Per-version audit / roast docs
artifacts/                 Local run outputs and pulled Kaggle bundles
benchmarks/                Benchmark registry and review inputs
```

## Quick Start

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m autoresearch_hv validate-version --version vR.1
python -m autoresearch_hv push-kaggle --version vR.1 --dry-run
```

## CLI

The primary entrypoint is:

```powershell
python -m autoresearch_hv <command> ...
```

Supported lifecycle commands:

- `scaffold-version`
- `validate-version`
- `push-kaggle`
- `kaggle-status`
- `pull-kaggle`
- `sync-run`
- `review-run`
- `mirror-obsidian`
- `next-ablation`

The shipped `vR.1` notebook delegates execution to repo modules under `src/autoresearch_hv/hndsr_vr/` and does not carry notebook-only model logic.
