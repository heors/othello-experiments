# MetaOthello Continuation Notes

These notes mirror the runnable commands in the root README. Run commands from
the repository root through Pixi.

```bash
pixi run download-classic-assets
```

`metaothello_helpers.curiosity_utils` is the maintained helper module.
`metaothello_helpers.curious_utils` is only a compatibility alias for older
imports.

## Evaluate A Checkpoint

Fresh random games:

```bash
pixi run python -m metaothello_helpers.eval_ckpt \
  --ckpt data/classic/ckpts/epoch_50.ckpt \
  --game classic \
  --num_games 2000 \
  --batch_size 1024
```

Zarr corpus evaluation:

```bash
pixi run python -m metaothello_helpers.eval_ckpt \
  --ckpt data/classic/ckpts/epoch_50.ckpt \
  --game classic \
  --zarr data/classic/train_classic_20M.zarr \
  --max_zarr_games 10000 \
  --batch_size 1024
```

This reports aggregate masked teacher-forced metrics over predicted moves 2..60
and also returns per-move arrays.

## Fixed-Corpus Resampling Continuation

```bash
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
```

The script samples a new subset from the 20M corpus every epoch, saves
`epoch_{N}.ckpt` under `run_dir/ckpts/`, and stores optimizer state in
`run_dir/training_state.pt`.

## Curious Continuation With kNN Novelty

```bash
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
```

Use `--refreshes_per_epoch 2` to regenerate the curious corpus every half
epoch.

## Curious Continuation With Count Novelty

```bash
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
```

## Standalone Curious Corpus Generation

kNN novelty:

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

Count novelty:

```bash
pixi run python -m metaothello_helpers.generate_curious_data \
  --ckpt data/classic/ckpts/epoch_50.ckpt \
  --out data/classic_curious_e50/train_curious_count_10k.zarr \
  --game classic \
  --num_games 10000 \
  --novelty count \
  --beta 0.25 \
  --tau 1.0
```

Add `--archive_out archive_state.npz` if you want to persist novelty state, and
add `--no_wandb` to training commands if you do not want Weights & Biases
logging.
