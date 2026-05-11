from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metaothello.mingpt.tokenizer import Tokenizer

from metaothello_helpers.curiosity_utils import (
    BLOCK_SIZE,
    ZarrSequenceSampler,
    configure_runtime,
    evaluate_random_games,
    evaluate_sequence_loss,
    generate_curious_corpus,
    get_device,
    load_archive,
    load_training_model,
    load_training_state,
    make_novelty_archive,
    make_optimizer_config,
    maybe_init_wandb,
    raw_model,
    resolve_autocast,
    save_archive,
    save_json,
    save_model_ckpt,
    save_training_state,
    train_one_pass,
    wandb_log,
    write_corpus_zarr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MetaOthello checkpoint with on-the-fly curious corpus refreshes.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--init_ckpt", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10, help="Number of continuation epochs on a fresh run.")
    parser.add_argument("--extend_epochs", type=int, default=0, help="Extra epochs to append to an existing run target.")
    parser.add_argument("--game", type=str, default="classic")
    parser.add_argument("--games_per_refresh", type=int, default=25_000)
    parser.add_argument("--refreshes_per_epoch", type=int, default=1, help="Use 2 for half-epoch corpus refreshes.")
    parser.add_argument("--novelty", type=str, choices=["knn", "count"], default="knn")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--archive_size", type=int, default=50_000)
    parser.add_argument("--persist_archive", action="store_true", default=True)
    parser.add_argument("--reset_archive_each_refresh", action="store_false", dest="persist_archive")
    parser.add_argument("--save_corpus_dir", type=Path, default=None)
    parser.add_argument("--val_zarr", type=Path, default=None)
    parser.add_argument("--val_games", type=int, default=10_000)
    parser.add_argument("--eval_num_games", type=int, default=2_000)
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
    parser.add_argument("--deterministic_first_move", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="metaothello")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_val_seqs(run_dir: Path, val_zarr: Path | None, val_games: int, seed: int) -> np.ndarray | None:
    if val_zarr is None or val_games <= 0:
        return None
    val_idx_path = run_dir / "val_indices.npy"
    val_idx = np.load(val_idx_path) if val_idx_path.exists() else None
    sampler = ZarrSequenceSampler(val_zarr, val_games=val_games, seed=seed, val_idx=val_idx)
    if not val_idx_path.exists():
        np.save(val_idx_path, sampler.val_idx)
    return sampler.load_val()


def build_refresh_payload(epoch: int, refresh_idx: int, train_stats: dict[str, Any], gen_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "refresh": refresh_idx,
        "train_loss": float(train_stats["train_loss"]),
        "lr": float(train_stats["last_lr"]),
        "tokens_seen": int(train_stats["tokens_seen"]),
        "num_steps": int(train_stats["num_steps"]),
        "num_examples": int(train_stats["num_examples"]),
        "generated_games": int(gen_stats["num_games"]),
        "gen_novelty_mean": float(gen_stats["novelty_mean"]),
        "gen_novelty_last": float(gen_stats["novelty_last"]),
        "archive_size": float(gen_stats["archive_size"]),
    }


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.run_dir / "ckpts"
    state_path = args.run_dir / "training_state.pt"
    archive_path = args.run_dir / "archive_state.npz"
    epoch_log_dir = args.run_dir / "epoch_logs"
    refresh_log_dir = args.run_dir / "refresh_logs"
    epoch_log_dir.mkdir(parents=True, exist_ok=True)
    refresh_log_dir.mkdir(parents=True, exist_ok=True)
    if args.save_corpus_dir is not None:
        args.save_corpus_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime(args.seed)
    device = get_device()
    autocast = resolve_autocast(device, amp_dtype=args.amp_dtype, enabled=args.amp_dtype != "none")
    tok = Tokenizer()
    val_seqs = load_val_seqs(args.run_dir, args.val_zarr, args.val_games, args.seed)

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

    total_planned_refreshes = max(1, (target_epoch - continuation_base_epoch) * args.refreshes_per_epoch)
    final_tokens = total_planned_refreshes * args.games_per_refresh * BLOCK_SIZE
    warmup_tokens = int(max(1.0, args.warmup_epochs * args.refreshes_per_epoch * args.games_per_refresh * BLOCK_SIZE))

    run = maybe_init_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_name or args.run_dir.name,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    if args.persist_archive:
        archive = load_archive(archive_path, args.novelty, args.archive_size)
    else:
        archive = make_novelty_archive(args.novelty, args.archive_size)

    if loaded_ckpt is not None:
        print(json.dumps({"loaded_ckpt": str(loaded_ckpt), "start_epoch": start_epoch, "target_epoch": target_epoch}, indent=2))

    steps_per_refresh_for_logging = max(1, math.ceil(args.games_per_refresh / args.batch_size))
    global_refresh = max(0, (start_epoch - continuation_base_epoch) * args.refreshes_per_epoch)
    for epoch in range(start_epoch + 1, target_epoch + 1):
        refresh_payloads: list[dict[str, Any]] = []

        for refresh_idx in range(args.refreshes_per_epoch):
            if not args.persist_archive:
                archive = make_novelty_archive(args.novelty, args.archive_size)

            seqs, boards, gen_stats = generate_curious_corpus(
                model,
                tok,
                archive,
                num_games=args.games_per_refresh,
                game_alias=args.game,
                beta=args.beta,
                tau=args.tau,
                k=args.k,
                seed_first_move_random=not args.deterministic_first_move,
                autocast=autocast,
                progress=True,
            )

            if args.save_corpus_dir is not None:
                corpus_name = f"epoch_{epoch:03d}_refresh_{refresh_idx:02d}_{args.novelty}.zarr"
                write_corpus_zarr(seqs, boards, args.save_corpus_dir / corpus_name)
                save_json(args.save_corpus_dir / f"epoch_{epoch:03d}_refresh_{refresh_idx:02d}_{args.novelty}.json", gen_stats)

            train_stats = train_one_pass(
                model,
                optimizer,
                seqs,
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
                log_callback=lambda payload, global_refresh=global_refresh: wandb_log(run, {f"train/{k}": v for k, v in payload.items()}, step=global_refresh * steps_per_refresh_for_logging + int(payload["step_in_pass"])),
            )
            tokens_seen = int(train_stats["tokens_seen"])

            refresh_payload = build_refresh_payload(epoch, refresh_idx, train_stats, gen_stats)
            refresh_payloads.append(refresh_payload)
            save_json(refresh_log_dir / f"epoch_{epoch:03d}_refresh_{refresh_idx:02d}.json", refresh_payload)
            refresh_log_step = (global_refresh + 1) * steps_per_refresh_for_logging
            wandb_log(run, {f"refresh/{k}": v for k, v in refresh_payload.items() if isinstance(v, (int, float))}, step=refresh_log_step)
            print(json.dumps(refresh_payload, indent=2))
            global_refresh += 1

        val_loss = None
        if val_seqs is not None:
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
        if args.persist_archive:
            save_archive(archive, archive_path)

        epoch_payload = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean([p["train_loss"] for p in refresh_payloads])),
            "mean_gen_novelty": float(np.mean([p["gen_novelty_mean"] for p in refresh_payloads])),
            "archive_size": float(refresh_payloads[-1]["archive_size"]),
            "lr": float(refresh_payloads[-1]["lr"]),
            "tokens_seen": int(tokens_seen),
        }
        if val_loss is not None:
            epoch_payload["val_loss"] = float(val_loss)
        if eval_metrics:
            epoch_payload.update({f"eval_{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))})

        save_training_state(
            state_path,
            {
                "epoch": epoch,
                "target_epoch": target_epoch,
                "continuation_base_epoch": continuation_base_epoch,
                "tokens_seen": tokens_seen,
                "optimizer_state_dict": optimizer.state_dict(),
                "latest_ckpt": str(ckpt_path),
                "persist_archive": args.persist_archive,
                "novelty": args.novelty,
            },
        )
        save_json(epoch_log_dir / f"epoch_{epoch}.json", epoch_payload)
        epoch_log_step = global_refresh * steps_per_refresh_for_logging
        wandb_log(run, epoch_payload, step=epoch_log_step)
        print(json.dumps(epoch_payload, indent=2))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
