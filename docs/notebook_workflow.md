# HNDSR Notebook Workflow

## Version Rules

- Scratch lineage notebooks use `vR.x_HNDSR.ipynb`.
- External pretrained lineage notebooks use `vR.P.x_HNDSR.ipynb`.
- A new notebook version is created only when the model family, optimizer/loss family, dataset/protocol, checkpoint source, or evaluation contract changes.
- Runtime-only Kaggle fixes stay in the same version until the review closes.

## Immutable Lifecycle

1. Scaffold the next notebook version and its paired markdown doc.
2. Run the local readiness validator before Kaggle handoff.
3. Commit and push the scaffold checkpoint to GitHub.
4. Run the notebook on Kaggle and return the executed notebook.
5. Sync the returned notebook into the run manifest before further fixes.
6. Write the paired review and roast doc, then commit and push again.
7. Fork the next version only after the current version is frozen.
