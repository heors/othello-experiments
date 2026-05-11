from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metaothello.analysis_utils import gen_games
from metaothello.mingpt.tokenizer import Tokenizer

from metaothello_helpers.curiosity_utils import (
    configure_runtime,
    evaluate_teacher_forced,
    get_device,
    load_eval_model,
    resolve_autocast,
    save_json,
    valid_masks_from_token_seqs,
)


def load_dataset_eval_data(
    zarr_path: Path,
    *,
    tok: Tokenizer,
    game_alias: str,
    num_games: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError("xarray with zarr support is required for --zarr evaluation") from exc

    ds = xr.open_zarr(zarr_path)
    n_total = int(ds.sizes["game"])

    if num_games is None or num_games <= 0 or num_games >= n_total:
        seqs = ds["seqs"].values.astype(np.int32)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=num_games, replace=False)).astype(np.int64)
        seqs = ds["seqs"].isel(game=idx).values.astype(np.int32)

    valid_masks = valid_masks_from_token_seqs(seqs, game_alias=game_alias, tok=tok)
    return seqs, valid_masks


@torch.inference_mode()
def evaluate_checkpoint(
    ckpt_path: Path,
    *,
    game_alias: str,
    num_games: int | None,
    batch_size: int,
    dataset_zarr: Path | None,
    seed: int,
    amp_dtype: str,
) -> dict[str, object]:
    device = get_device()
    autocast = resolve_autocast(device, amp_dtype=amp_dtype, enabled=amp_dtype != "none")
    tok = Tokenizer()

    if dataset_zarr is None:
        if num_games is None or num_games <= 0:
            raise ValueError("num_games must be positive when evaluating on fresh random games")
        seqs, valid_masks = gen_games(game_alias, num_games=num_games, tokenizer=tok)
        source = f"fresh_random:{game_alias}"
    else:
        seqs, valid_masks = load_dataset_eval_data(
            dataset_zarr,
            tok=tok,
            game_alias=game_alias,
            num_games=num_games,
            seed=seed,
        )
        source = str(dataset_zarr)

    model = load_eval_model(ckpt_path, device)
    metrics = evaluate_teacher_forced(
        model,
        seqs,
        valid_masks,
        batch_size=batch_size,
        device=device,
        autocast=autocast,
    )
    metrics.update(
        {
            "ckpt": str(ckpt_path),
            "game": game_alias,
            "num_games": int(seqs.shape[0]),
            "source": source,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MetaOthello checkpoint over all next-token positions under teacher forcing. "
            "Works on fresh random games or a Zarr corpus."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--game", "--game_alias", dest="game", type=str, default="classic")
    parser.add_argument(
        "--num_games",
        "--max_zarr_games",
        dest="num_games",
        type=int,
        default=10_000,
        help="Number of fresh games, or max number of sampled Zarr games. Use <=0 with --zarr to evaluate the whole corpus.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--zarr", "--dataset_zarr", dest="dataset_zarr", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--stats_out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.seed)
    metrics = evaluate_checkpoint(
        args.ckpt,
        game_alias=args.game,
        num_games=args.num_games,
        batch_size=args.batch_size,
        dataset_zarr=args.dataset_zarr,
        seed=args.seed,
        amp_dtype=args.amp_dtype,
    )
    if args.stats_out is not None:
        save_json(args.stats_out, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
