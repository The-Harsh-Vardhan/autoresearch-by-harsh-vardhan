# Versioned Kaggle Notebooks

- `vR.x_HNDSR.ipynb` is reserved for scratch-trained notebook versions.
- `vR.P.x_HNDSR.ipynb` is reserved for externally pretrained notebook versions.
- Do not overwrite a reviewed notebook version.
- Pair every notebook with:
  - `docs/notebooks/<stem>.md`
  - `reports/reviews/<stem>.review.md`
- Run `python -m autoresearch_hv validate-version --version <version>` before handing a notebook to Kaggle.
