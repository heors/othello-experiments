#!/usr/bin/env bash
#SBATCH --job-name=mo_fixed_e50
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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

# Match the curious runs' total examples per epoch: 25k x 2 refreshes = 50k.
pixi run python -m metaothello_helpers.train_fixed_resample \
  --run_dir data/classic_fixed_e50 \
  --src_zarr data/classic/train_classic_20M.zarr \
  --init_ckpt data/classic/ckpts/epoch_50.ckpt \
  --epochs 10 \
  --games_per_epoch 5000 \
  --val_games 10000 \
  --eval_num_games 2000 \
  --game classic \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name classic_fixed_e50
