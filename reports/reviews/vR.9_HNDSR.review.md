# vR.9 HNDSR Satellite Super-Resolution Review

## Status

- State: reviewed
- Domain: HNDSR Satellite Super-Resolution

## Run Intake

- Returned notebook path: C:\D Drive\Projects\My Learnings\Auto Research Learning\artifacts\test-fixtures\lifecycle-root-0578da0e\incoming\kaggle-output\vR.9_HNDSR.ipynb
- Best checkpoint path: C:\D Drive\Projects\My Learnings\Auto Research Learning\artifacts\test-fixtures\lifecycle-root-0578da0e\incoming\kaggle-output\vR.9_train_best.pt
- Full evaluation summary path: C:\D Drive\Projects\My Learnings\Auto Research Learning\artifacts\test-fixtures\lifecycle-root-0578da0e\incoming\kaggle-output\eval_summary.json
- W&B URL: https://wandb.example/run/vr9

## Findings

- [LOW] PSNR exceeds the bicubic control by 0.5961.
- [LOW] SSIM exceeds the bicubic control by 0.0435.

## Roast

- The platform is not allowed to hide behind a pretty notebook if the pulled bundle is incomplete.
- Any run without an online W&B URL is operationally unfinished, not almost done.
- If bicubic still wins, architecture heroics are premature; the first fix is discipline, not bravado.

## Promotion Decision

- Decision: freeze and fork next version
- Next bounded ablations:
- vR.9.1: increase model_channels from 32 to 64 while keeping the dataset and evaluation contract fixed
- vR.10: raise inference_steps from 10 to 20 without changing the training dataset or loss family
