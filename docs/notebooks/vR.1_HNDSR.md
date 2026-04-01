# vR.1 HNDSR

## Objective

`vR.1` is the first immutable Kaggle notebook in the scratch lineage for the standalone AutoResearch repo. Its job is to prove the repo-owned lifecycle, not to overstate model quality.

## Kaggle Run Guide

1. Open `notebooks/versions/vR.1_HNDSR.ipynb`.
2. Run the runtime diagnostics cells first and confirm CUDA visibility if a GPU runtime is enabled.
3. Run the validation cell before any training cell.
4. Run the bicubic control evaluation to confirm the dataset and metrics path.
5. Run the smoke training path to confirm checkpoint and evaluation wiring.
6. Run the full training and full evaluation cells only after the smoke path succeeds.
7. Pull the executed notebook and outputs back into the repo before review.

## Config Contract

- Full training config: `configs/hndsr_vr/vR.1_train.yaml`
- Smoke training config: `configs/hndsr_vr/vR.1_smoke.yaml`
- Bicubic control config: `configs/hndsr_vr/vR.1_control.yaml`
- Fixed scale: `4x`
- W&B mode: `online` for milestone completion

## Expected Artifacts

- Control eval summary JSON under `artifacts/vR.1-control/metrics/`
- Smoke checkpoint and metrics under `artifacts/vR.1-smoke/`
- Full training checkpoint and metrics under `artifacts/vR.1-train/`
- Full evaluation summary and image strips under `artifacts/vR.1-eval/`
- Tracker records under each run's `tracker/` directory

## Handoff Back For Review

Return all of the following after the Kaggle run:

- The executed `vR.1_HNDSR.ipynb`
- Any Kaggle-side edits required for runtime stability
- The best checkpoint path used for evaluation
- The control, smoke, and full evaluation JSON summaries
- The comparison grid image path
- The W&B run URL
- A short note about runtime duration, GPU type, and any failure modes hit during the run
