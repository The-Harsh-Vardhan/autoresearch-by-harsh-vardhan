# NLP Language Modelling Research Program

This file is the human-owned research program for the NLP LM domain.

## Objective

Train and improve character-level language models through bounded, reviewable experiments that track progress via bits-per-byte (BPB) on a fixed validation set.

## Fixed Contracts

- Dataset and tokenizer (character-level) are fixed per version.
- Evaluation metric is BPB on the held-out validation split.
- Every frozen version must have a notebook, an external doc, and a review/roast doc.

## Allowed Mutation Surface

- Model architecture changes (depth, width, attention pattern)
- Optimizer and learning rate schedule changes
- Batch size and training duration changes
- New model families (e.g., switching from GPT-nano to a different architecture)

## Disallowed Mutation Surface

- Changing the validation split after a version is frozen
- Ad hoc notebook-only training logic
- Untracked experiments without W&B logging

## Version Rules

- `vN.M` is a major step: model family, dataset, or evaluation contract changed.
- `vN.M.P` is a minor bounded change within the same lane.

## Engineering Baseline

Apply Power-of-10 discipline conservatively:

- Keep control flow simple.
- Bound loops, retries, and polling.
- Keep runtime state scoped and explicit.
- Validate inputs and check fallible returns.
