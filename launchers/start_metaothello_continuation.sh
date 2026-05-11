#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. This launcher bootstraps the classic assets,
# evaluates the shared epoch-50 checkpoint, and starts the three continuation
# branches used by the experiment.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

pixi run download-classic-assets

pixi run python -m metaothello_helpers.eval_ckpt \
  --ckpt data/classic/ckpts/epoch_50.ckpt \
  --game classic \
  --num_games 2000 \
  --batch_size 1024

pixi run python -m metaothello_helpers.train_fixed_resample \
  --run_dir data/classic_fixed_e50 \
  --src_zarr data/classic/train_classic_20M.zarr \
  --init_ckpt data/classic/ckpts/epoch_50.ckpt \
  --epochs 10 \
  --games_per_epoch 100000 \
  --val_games 10000 \
  --eval_num_games 2000 \
  --batch_size 4096 \
  --learning_rate 1e-4

pixi run python -m metaothello_helpers.train_curious_resample \
  --run_dir data/classic_curious_knn_e50 \
  --init_ckpt data/classic/ckpts/epoch_50.ckpt \
  --epochs 10 \
  --game classic \
  --novelty knn \
  --beta 0.25 \
  --tau 1.0 \
  --k 32 \
  --archive_size 50000 \
  --games_per_refresh 25000 \
  --refreshes_per_epoch 2 \
  --val_zarr data/classic/train_classic_20M.zarr \
  --val_games 10000 \
  --eval_num_games 2000 \
  --batch_size 4096 \
  --learning_rate 1e-4

pixi run python -m metaothello_helpers.train_curious_resample \
  --run_dir data/classic_curious_count_e50 \
  --init_ckpt data/classic/ckpts/epoch_50.ckpt \
  --epochs 10 \
  --game classic \
  --novelty count \
  --beta 0.25 \
  --tau 1.0 \
  --archive_size 50000 \
  --games_per_refresh 25000 \
  --refreshes_per_epoch 2 \
  --val_zarr data/classic/train_classic_20M.zarr \
  --val_games 10000 \
  --eval_num_games 2000 \
  --batch_size 4096 \
  --learning_rate 1e-4
