#!/usr/bin/env bash
#SBATCH --job-name=mo_curious_knn_e50
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required on the compute node for this launcher." >&2
  exit 1
fi

if [ -f "${HOME}/.secrets/wandb.sh" ]; then
  source "${HOME}/.secrets/wandb.sh"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-metaothello}"
export PYTHONUNBUFFERED=1

pixi run python -m metaothello_helpers.train_curious_resample \
  --run_dir data/classic_curious_knn_e50 \
  --init_ckpt data/classic/ckpts/epoch_50.ckpt \
  --epochs 10 \
  --game classic \
  --novelty knn \
  --beta 0.25 \
  --tau 1.0 \
  --k 32 \
  --archive_size 10000 \
  --games_per_refresh  2500 \
  --refreshes_per_epoch 2 \
  --val_zarr data/classic/train_classic_20M.zarr \
  --val_games 10000 \
  --eval_num_games 2000 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name classic_curious_knn_e50
