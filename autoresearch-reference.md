# `karpathy/autoresearch` Reference Guide

Based on the upstream `master` branch as inspected on March 28, 2026.

Repo: https://github.com/karpathy/autoresearch

## What This Repo Is

`autoresearch` is a deliberately small benchmark for autonomous model research.

The core idea is simple:

- The human writes the research organization in `program.md`.
- The agent edits only `train.py`.
- `prepare.py` stays fixed so experiments remain comparable.

This repo is not trying to be a general training framework. It is trying to make one question easy to test:

> What changes to a single-file training script improve validation quality within a fixed 5-minute training budget?

## The Three Important Files

### `prepare.py`

This file defines the fixed world:

- data download
- tokenizer training and loading
- dataloader construction
- evaluation metric
- hard constraints like context length and time budget

This file is intentionally read-only during research.

### `train.py`

This is the only file the agent is supposed to modify.

It contains:

- model architecture
- optimizer definitions
- training loop
- hyperparameters
- schedules
- final evaluation and summary output

If the repo is the lab, `train.py` is the bench.

### `program.md`

This is the operating manual for the autonomous researcher.

It defines:

- how to start a run
- what the agent is allowed to edit
- how to log results
- how to decide keep vs discard
- how the loop continues without waiting for the user

If `train.py` defines the model, `program.md` defines the organization that improves it.

## Mental Model

The repo works because it separates three concerns cleanly:

1. Benchmark contract: `prepare.py`
2. Research surface: `train.py`
3. Research process: `program.md`

That separation matters. If the agent could change the evaluation harness, dataset split, tokenizer rules, or timing rules, it could "improve" the result by changing the benchmark instead of improving the model.

## Fixed Benchmark Contract in `prepare.py`

These are the most important invariants:

- `MAX_SEQ_LEN = 2048`
- `TIME_BUDGET = 300`
- `EVAL_TOKENS = 40 * 524288`
- pinned validation shard: `shard_06542.parquet`
- cache root: `~/.cache/autoresearch/`
- tokenizer vocab size: `VOCAB_SIZE = 8192`

### Why these are fixed

The benchmark is designed to compare ideas under one stable setup:

- same sequence length
- same wall-clock training budget
- same validation data
- same metric
- same tokenizer pipeline

This gives the agent freedom to change the model and optimizer while preserving scientific comparability.

### Why wall-clock time is fixed

The repo fixes **time**, not steps or epochs.

That is intentional because the agent may change:

- model size
- batch size
- attention pattern
- optimizer cost
- throughput

If the benchmark fixed steps, faster models and slower models would not be compared fairly. Fixing 5 minutes asks the practical question:

> What learns best on this machine in 5 minutes?

## Data and Evaluation Pipeline

`prepare.py` does two jobs: one-time preparation and runtime support for `train.py`.

### One-time preparation

1. Download parquet shards from the Hugging Face dataset.
2. Train a BPE tokenizer with `rustbpe`.
3. Save the tokenizer and token-byte lookup into the cache.

### Runtime support

`train.py` imports these symbols from `prepare.py`:

- `MAX_SEQ_LEN`
- `TIME_BUDGET`
- `Tokenizer`
- `make_dataloader`
- `evaluate_bpb`

This is the main contract between the fixed harness and the editable training script.

### Batch flow, end to end

One batch moves through the system like this:

1. A parquet file is read from `~/.cache/autoresearch/data/`.
2. Text documents are extracted from the `text` column.
3. The tokenizer encodes them and prepends BOS.
4. `make_dataloader(...)` packs documents into fixed-length rows with best-fit packing.
5. It yields `inputs` and `targets` tensors for next-token prediction.
6. `train.py` computes cross-entropy during training.
7. `evaluate_bpb(...)` evaluates the trained model on the fixed validation shard.

### Why `val_bpb` is used

The key metric is validation bits per byte:

- lower is better
- it is vocab-size independent
- it stays comparable even if architecture details change

This makes it a better benchmark metric than raw token-level loss when tokenization choices matter.

## `train.py` Walkthrough

`train.py` is organized like a compact research script, not a framework.

### 1. Environment and kernel setup

At startup the script:

- sets CUDA allocator options
- disables HF progress bars
- loads a Flash Attention 3 kernel through `kernels`
- chooses the kernel repo based on GPU capability

This is performance plumbing, not the main research surface.

### 2. Model config

`GPTConfig` contains the main structural fields:

- `sequence_len`
- `vocab_size`
- `n_layer`
- `n_head`
- `n_kv_head`
- `n_embd`
- `window_pattern`

The editable high-level knobs later in the file drive these values:

- `DEPTH`
- `ASPECT_RATIO`
- `HEAD_DIM`
- `WINDOW_PATTERN`

`build_model_config(depth)` computes:

- `model_dim = depth * ASPECT_RATIO`, rounded to a multiple of `HEAD_DIM`
- `num_heads = model_dim // HEAD_DIM`

This means model size is mostly controlled through a few direct constants.

### 3. Transformer blocks

The model is a compact GPT-style stack:

- token embedding
- repeated `Block`s
- final norm
- linear language modeling head

Each `Block` contains:

- causal self-attention
- MLP
- residual updates around both

Normalization is RMS norm via `F.rms_norm`.

### 4. Attention details

`CausalSelfAttention` builds:

- query projection
- key projection
- value projection
- output projection

Repo-specific details that matter:

- grouped query / key-value structure exists through `n_head` and `n_kv_head`
- rotary position embedding is applied to `q` and `k`
- attention windows are controlled per layer through `WINDOW_PATTERN`
- some layers receive value embeddings mixed into `v` through a learned gate

`WINDOW_PATTERN` uses:

- `L` for full context
- `S` for half-context sliding attention

The last layer is always forced to full context.

### 5. Value embeddings and residual mixing

This repo is not a plain minimal transformer.

It includes:

- per-layer value embeddings on alternating layers
- learned scalar mixing through `resid_lambdas`
- learned skip-from-input mixing through `x0_lambdas`

These are part of the research surface. They are exactly the sort of architectural choices an agent can simplify, remove, or tune.

### 6. MLP

The MLP is intentionally simple:

1. linear up projection
2. `relu().square()`
3. linear down projection

It is not using a more elaborate gated MLP here. That simplicity is consistent with the repo's design goal: compact, editable, and easy to diff.

### 7. Parameter accounting and FLOPs

The script reports parameter counts by category:

- token embeddings
- value embeddings
- language modeling head
- transformer matrices
- scalar parameters

It also estimates FLOPs per token.

These numbers feed the summary and help interpret throughput and MFU.

### 8. Optimizer split

This is one of the most important ideas in the file.

The repo uses two optimizer families:

- `Muon` for 2D matrix parameters
- `AdamW` for embeddings, unembedding, and scalar parameters

Why this matters:

- different parameter types can benefit from different optimization behavior
- much of the repo's experimental surface is in how these parameter groups are tuned

The key learning-rate knobs are:

- `EMBEDDING_LR`
- `UNEMBEDDING_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `WEIGHT_DECAY`
- `ADAM_BETAS`

### 9. Training budget and batching

These constants are the main throughput levers:

- `TOTAL_BATCH_SIZE`
- `DEVICE_BATCH_SIZE`

The script computes:

- tokens per forward/backward pass = `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`
- gradient accumulation steps = `TOTAL_BATCH_SIZE // tokens_per_fwdbwd`

So the model can simulate a large effective batch even if the device batch is smaller.

This is a major research tradeoff surface:

- larger total batch can improve optimization stability
- larger device batch can improve utilization
- larger settings can also increase VRAM and reduce step frequency

### 10. Scheduling

The learning-rate schedule is time-based, not step-based.

Important constants:

- `WARMUP_RATIO`
- `WARMDOWN_RATIO`
- `FINAL_LR_FRAC`

`progress` is defined from elapsed training time divided by `TIME_BUDGET`.

This means schedules stay aligned to the fixed 5-minute benchmark even if changes affect step speed.

### 11. Training loop

The loop is structured around the time budget:

1. fetch batches
2. run gradient accumulation
3. update schedules from time progress
4. step the optimizer
5. zero gradients
6. log throughput and utilization
7. stop once training time reaches 300 seconds, after warmup/compile overhead

There is also a fast-fail condition:

- if loss becomes `NaN`
- or if loss exceeds 100

the script prints `FAIL` and exits.

### 12. Final evaluation and summary

After training:

- model switches to eval mode
- `evaluate_bpb(...)` runs on validation data
- the script prints a summary block

The key outputs are:

- `val_bpb`
- `training_seconds`
- `total_seconds`
- `peak_vram_mb`
- `mfu_percent`
- `total_tokens_M`
- `num_steps`
- `num_params_M`
- `depth`

## `program.md` As The Autonomous Research Protocol

`program.md` is not just a note. It is the human-authored control layer for the research loop.

### Setup rules

The agent is instructed to:

1. choose a fresh run tag
2. create a new branch like `autoresearch/<tag>`
3. read `README.md`, `prepare.py`, and `train.py`
4. verify the cache exists
5. initialize `results.tsv`
6. establish a baseline run before making changes

### Allowed and forbidden changes

The allowed scope is intentionally narrow:

- edit `train.py`

The forbidden scope is explicit:

- do not edit `prepare.py`
- do not install packages
- do not change the evaluation harness

This forces the agent to do actual model/training research, not environment hacking.

### Experiment loop

The intended loop is:

1. inspect current git state
2. modify `train.py` with one idea
3. commit
4. run `uv run train.py > run.log 2>&1`
5. extract `val_bpb` and `peak_vram_mb`
6. log to `results.tsv`
7. keep the commit if it improved
8. revert if it did not
9. continue indefinitely

### Logging

`results.tsv` uses these columns:

- `commit`
- `val_bpb`
- `memory_gb`
- `status`
- `description`

Status values:

- `keep`
- `discard`
- `crash`

The TSV is intentionally left untracked by git.

### Crash policy

If a run crashes:

- easy, accidental bugs should be fixed and rerun
- fundamentally bad ideas should be logged as `crash` and skipped

### Why this structure works

The repo combines:

- a fixed benchmark
- a narrow editable surface
- a repeatable research policy

That makes it unusually easy for an autonomous coding agent to perform overnight hill-climbing on a real training script.

## What The Human Programs vs What The Agent Programs

This distinction is central to understanding the repo.

### Human-programmed org

The human writes the operating system for the research process in `program.md`:

- naming runs
- branch discipline
- logging discipline
- keep/discard policy
- autonomy rules

### Agent-programmed model

The agent edits the object being researched in `train.py`:

- model architecture
- optimizer behavior
- schedules
- batch sizes
- structural simplifications or additions

This is the repo's real thesis:

> The human increasingly writes the meta-process, while the agent iterates on the model code itself.

## Safe Experiment Categories in `train.py`

If you were reviewing an agent's proposed changes, these are the major categories that make sense.

### 1. Architecture

Examples:

- change `DEPTH`
- change `WINDOW_PATTERN`
- simplify or remove value embeddings
- change head structure
- simplify the MLP

### 2. Optimizer behavior

Examples:

- tune `MATRIX_LR`
- tune `EMBEDDING_LR`
- tune `ADAM_BETAS`
- change Muon momentum behavior
- adjust weight decay

### 3. Batching and throughput

Examples:

- change `TOTAL_BATCH_SIZE`
- change `DEVICE_BATCH_SIZE`
- trade utilization against memory

### 4. Parameter count and shape

Examples:

- alter `ASPECT_RATIO`
- alter `HEAD_DIM`
- shrink or expand the model under the same time budget

### 5. Schedule

Examples:

- introduce warmup
- change warmdown duration
- leave a nonzero final LR fraction

## What Is Intentionally Out Of Bounds

The following are not supposed to be part of experimentation:

- editing `prepare.py`
- changing `evaluate_bpb`
- changing the validation split
- adding dependencies
- broadening the codebase into a multi-file system

If those change, the benchmark itself changes.

## How To Read Results

### `val_bpb`

Primary metric. Lower is better.

### `peak_vram_mb`

Memory usage. Important as a soft constraint.

Higher memory may be acceptable if the quality gain is meaningful, but massive memory blowups are usually bad tradeoffs.

### `mfu_percent`

A utilization estimate. Useful for understanding how well the run is using the hardware, but not the primary goal.

### `total_tokens_M`

How many tokens were processed in the fixed time budget.

This helps explain why a smaller or faster model may win even if it is less expressive per step.

### `num_params_M`

Model size. Useful for interpreting scaling tradeoffs.

## Recommended Reading Order

If you revisit the repo later, this is the best order:

1. `README.md`
2. `program.md`
3. `prepare.py`
4. `train.py`

That order keeps the big picture clear:

- what the repo is for
- how the autonomous loop is supposed to behave
- what the fixed contract is
- what the agent is actually allowed to change

## Questions To Check Your Understanding

Use these as self-checks when you review the repo again:

1. Why does the benchmark fix wall-clock time instead of steps?
2. Why is `prepare.py` read-only?
3. What is the contract from `prepare.py` into `train.py`?
4. How does one batch move from parquet text to model loss?
5. Why is `val_bpb` a better cross-run metric here than raw loss alone?
6. What kinds of experiments belong in `train.py`, and what kinds do not?
7. What is the difference between `program.md` and `train.py` in the overall system?

## Short Takeaway

`autoresearch` is best understood as a constrained autonomous research loop:

- fixed benchmark
- one editable training file
- one human-authored research protocol
- repeated 5-minute experiments
- keep only improvements

The code is small on purpose. The interesting part is not scale. The interesting part is the separation between benchmark, model, and research process.
