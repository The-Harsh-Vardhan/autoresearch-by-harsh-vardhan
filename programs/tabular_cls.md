# Tabular Classification Research Program

## Objective

Prove the domain-agnostic architecture by implementing a third research domain
focused on tabular classification using standard ML benchmark datasets (Iris, Titanic).

## Datasets

| Dataset | Rows | Features | Classes | Task |
|---------|------|----------|---------|------|
| Iris    | 150  | 4        | 3       | Flower species classification |
| Titanic | 891  | 9        | 2       | Survival prediction |

## Models

- **Logistic Baseline**: Single-layer logistic regression (control)
- **SmallMLP**: Two hidden-layer MLP with ReLU and dropout

## Primary Metric

**Accuracy** (higher_is_better)

## Research Questions

1. Does the MLP beat logistic regression on both datasets?
2. Does increasing `hidden_dim` from 64 → 128 help?
3. Does the Titanic domain benefit from more feature engineering?

## Version Contract

Each version must produce:
- Notebook, doc, review, and config triad (control / smoke / train)
- Real baseline numbers in the benchmark registry
- A post-run review with findings and ablation suggestions
