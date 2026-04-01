# vR.1 HNDSR Review

## Status

- State: reviewed
- Lineage: scratch (`vR.x`)
- Scope: immutable SR3 Kaggle notebook for the standalone AutoResearch repo

## Run Intake

- Returned notebook path: artifacts\test-fixtures\kaggle-output-d1ba8fb0\vR.1_HNDSR.ipynb
- Best checkpoint path: artifacts\test-fixtures\kaggle-output-d1ba8fb0\vR.1_train_best.pt
- Control summary path: artifacts/vR.1-control/metrics/eval_summary.json
- Smoke summary path: artifacts/vR.1-smoke-eval/metrics/eval_summary.json
- Full evaluation summary path: artifacts\test-fixtures\kaggle-output-d1ba8fb0\eval_summary.json
- W&B URL: https://wandb.example/run/vr1

## Audit Checklist

- Notebook structure stayed aligned with the contract.
- Repo modules, not notebook cells, owned train/eval logic.
- Metrics, checkpoint, and sample artifact paths are traceable.
- W&B tracker state is explicit.
- Kaggle-only fixes are mirrored back into repo code or docs.

## Findings

- [LOW] PSNR exceeds the bicubic control by 0.5961.
- [LOW] SSIM exceeds the bicubic control by 0.0435.

## Roast

- The platform is not allowed to hide behind a pretty notebook if the pulled bundle is incomplete.
- Any run without an online W&B URL is operationally unfinished, not almost done.
- If bicubic still wins, architecture heroics are premature; the first fix is discipline, not bravado.

## Promotion Decision

- Decision: freeze and fork next version
- Benchmark delta vs bicubic: PSNR +0.5961, SSIM +0.0435
- Next bounded ablations:
- vR.1.1: increase model_channels from 32 to 64 while keeping the dataset and evaluation contract fixed
- vR.2: raise inference_steps from 10 to 20 without changing the training dataset or loss family
