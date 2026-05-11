from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metaothello_helpers.curiosity_utils import (
    BLOCK_SIZE,
    ZarrSequenceSampler,
    configure_runtime,
    evaluate_random_games,
    evaluate_sequence_loss,
    get_device,
    load_training_model,
    load_training_state,
    make_optimizer_config,
    maybe_init_wandb,
    raw_model,
    resolve_autocast,
    save_json,
    save_model_ckpt,
    save_training_state,
    train_one_pass,
    wandb_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MetaOthello checkpoint with per-epoch fixed-corpus resampling.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--src_zarr", type=Path, required=True)
    parser.add_argument("--init_ckpt", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10, help="Number of continuation epochs on a fresh run.")
    parser.add_argument("--extend_epochs", type=int, default=0, help="Extra epochs to append to an existing run target.")
    parser.add_argument("--games_per_epoch", type=int, default=100_000)
    parser.add_argument("--val_games", type=int, default=10_000)
    parser.add_argument("--eval_num_games", type=int, default=2_000)
    parser.add_argument("--game", type=str, default="classic")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay", action="store_true", default=True)
    parser.add_argument("--no_lr_decay", action="store_false", dest="lr_decay")
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--wandb_project", type=str, default="metaothello")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_sampler(run_dir: Path, src_zarr: Path, val_games: int, seed: int) -> ZarrSequenceSampler:
    val_idx_path = run_dir / "val_indices.npy"
    if val_idx_path.exists():
        val_idx = np.load(val_idx_path)
    else:
        val_idx = None
    sampler = ZarrSequenceSampler(src_zarr, val_games=val_games, seed=seed, val_idx=val_idx)
    if not val_idx_path.exists():
        np.save(val_idx_path, sampler.val_idx)
    return sampler


def build_epoch_payload(epoch: int, train_stats: dict[str, Any], val_loss: float, eval_metrics: dict[str, Any], sample_seed: int) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "train_loss": float(train_stats["train_loss"]),
        "val_loss": float(val_loss),
        "lr": float(train_stats["last_lr"]),
        "tokens_seen": int(train_stats["tokens_seen"]),
        "num_steps": int(train_stats["num_steps"]),
        "num_examples": int(train_stats["num_examples"]),
        "sample_seed": int(sample_seed),
    }
    if eval_metrics:
        payload.update({f"eval_{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
    return payload


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.run_dir / "ckpts"
    state_path = args.run_dir / "training_state.pt"
    epoch_log_dir = args.run_dir / "epoch_logs"
    epoch_log_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime(args.seed)
    device = get_device()
    autocast = resolve_autocast(device, amp_dtype=args.amp_dtype, enabled=args.amp_dtype != "none")

    sampler = make_sampler(args.run_dir, args.src_zarr, args.val_games, args.seed)
    val_seqs = sampler.load_val()

    model, start_epoch, loaded_ckpt = load_training_model(
        init_ckpt=args.init_ckpt,
        ckpt_dir=ckpt_dir,
        device=device,
        data_parallel=True,
    )
    optimizer = raw_model(model).configure_optimizers(make_optimizer_config(args))

    training_state = load_training_state(state_path)
    if training_state is not None and int(training_state.get("epoch", -1)) == start_epoch:
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        tokens_seen = int(training_state.get("tokens_seen", 0))
        continuation_base_epoch = int(training_state.get("continuation_base_epoch", start_epoch))
        target_epoch = int(training_state.get("target_epoch", start_epoch))
    else:
        tokens_seen = 0
        continuation_base_epoch = start_epoch
        target_epoch = start_epoch + args.epochs

    target_epoch += int(args.extend_epochs)
    if start_epoch >= target_epoch:
        print(json.dumps({"status": "nothing_to_do", "start_epoch": start_epoch, "target_epoch": target_epoch}, indent=2))
        return

    total_planned_epochs = max(1, target_epoch - continuation_base_epoch)
    final_tokens = total_planned_epochs * args.games_per_epoch * BLOCK_SIZE
    warmup_tokens = int(max(1.0, args.warmup_epochs * args.games_per_epoch * BLOCK_SIZE))

    run = maybe_init_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_name or args.run_dir.name,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    if loaded_ckpt is not None:
        print(json.dumps({"loaded_ckpt": str(loaded_ckpt), "start_epoch": start_epoch, "target_epoch": target_epoch}, indent=2))

    steps_per_epoch_for_logging = max(1, math.ceil(args.games_per_epoch / args.batch_size))

    for epoch in range(start_epoch + 1, target_epoch + 1):
        sample_seed = args.seed + epoch
        train_seqs, _sample_idx = sampler.sample_train(args.games_per_epoch, seed=sample_seed)

        step_offset = (epoch - continuation_base_epoch - 1) * steps_per_epoch_for_logging

        train_stats = train_one_pass(
            model,
            optimizer,
            train_seqs,
            batch_size=args.batch_size,
            device=device,
            tokens_seen=tokens_seen,
            base_lr=args.learning_rate,
            warmup_tokens=warmup_tokens,
            final_tokens=final_tokens,
            lr_decay=args.lr_decay,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            autocast=autocast,
            log_interval=25,
            log_callback=lambda payload, step_offset=step_offset: wandb_log(run, {f"train/{k}": v for k, v in payload.items()}, step=step_offset + int(payload["step_in_pass"])),
        )
        tokens_seen = int(train_stats["tokens_seen"])

        val_loss = evaluate_sequence_loss(
            model,
            val_seqs,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            autocast=autocast,
        )

        eval_metrics: dict[str, Any] = {}
        if args.eval_num_games > 0:
            eval_metrics = evaluate_random_games(
                model,
                game_alias=args.game,
                num_games=args.eval_num_games,
                batch_size=args.batch_size,
                device=device,
                autocast=autocast,
            )

        ckpt_path = save_model_ckpt(model, ckpt_dir, epoch)
        save_training_state(
            state_path,
            {
                "epoch": epoch,
                "target_epoch": target_epoch,
                "continuation_base_epoch": continuation_base_epoch,
                "tokens_seen": tokens_seen,
                "optimizer_state_dict": optimizer.state_dict(),
                "latest_ckpt": str(ckpt_path),
            },
        )

        epoch_payload = build_epoch_payload(epoch, train_stats, val_loss, eval_metrics, sample_seed)
        save_json(epoch_log_dir / f"epoch_{epoch}.json", epoch_payload)
        epoch_log_step = step_offset + int(train_stats["num_steps"])
        wandb_log(run, epoch_payload, step=epoch_log_step)
        print(json.dumps(epoch_payload, indent=2))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
