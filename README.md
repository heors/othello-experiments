# MetaOthello Continuation Experiments

This is the experiment-only MetaOthello continuation repo. It keeps the code
needed to evaluate the shared Classic checkpoint, continue it on fixed data, and
continue it with novelty-biased curious self-play.

All commands are run through Pixi from the repository root.

## Setup

Install Pixi, then let Pixi create the environment on first run:

```bash
pixi run python --version
```

Download the Classic assets used by the continuation scripts:

```bash
pixi run download-classic-assets
```

Expected asset layout:

```text
data/classic/
  train_classic_20M.zarr/
  ckpts/
    epoch_50.ckpt
```

## Core Commands

Evaluate the epoch-50 checkpoint on fresh Classic games:

```bash
pixi run eval-classic-e50
```

Run the fixed-corpus continuation:

```bash
pixi run train-fixed-e50
```

Run curious continuation with kNN novelty:

```bash
pixi run train-curious-knn-e50
```

Run curious continuation with count-based novelty:

```bash
pixi run train-curious-count-e50
```

Generate a standalone curious corpus:

```bash
pixi run python -m metaothello_helpers.generate_curious_data \
  --ckpt data/classic/ckpts/epoch_50.ckpt \
  --out data/classic_curious_e50/train_curious_knn_10k.zarr \
  --game classic \
  --num_games 10000 \
  --novelty knn \
  --beta 0.25 \
  --tau 1.0 \
  --k 32
```

## Launchers

Slurm launchers live in `launchers/`:

```bash
sbatch launchers/train_fixed_e50.sh
sbatch launchers/train_curious_knn_e50.sh
sbatch launchers/train_curious_count_e50.sh
```

`launchers/start_metaothello_continuation.sh` runs the asset download, baseline
evaluation, and all three continuation branches in sequence. `launchers/run.sh`
is a small fixed-run convenience launcher.

## Paper Helpers

Interpretability and plotting helpers live in `paper/`.

```bash
pixi run python -m paper.curiosity_interp_starter --help
pixi run python -m paper.openended_othello_paperplots --help
```

Examples:

```bash
pixi run python -m paper.openended_othello_paperplots training-curves \
  --run fixed data/classic_fixed_e50 \
  --run curious data/classic_curious_knn_e50 \
  --summary_out data/paper/training_curves.json \
  --plot_out data/paper/training_curves.png
```

```bash
pixi run python -m paper.curiosity_interp_starter compare \
  --ckpt_a data/classic_fixed_e50/ckpts/epoch_60.ckpt \
  --ckpt_b data/classic_curious_knn_e50/ckpts/epoch_60.ckpt \
  --game classic \
  --num_games 1000 \
  --out_dir data/paper/interp_compare
```

## Useful Help

```bash
pixi run help-eval
pixi run help-generate-curious
pixi run help-train-fixed
pixi run help-train-curious
pixi run help-interp
pixi run help-paper-plots
```
