from __future__ import annotations

import argparse
import json
from pathlib import Path

from metaothello.mingpt.tokenizer import Tokenizer

from metaothello_helpers.curiosity_utils import (
    configure_runtime,
    generate_curious_corpus,
    get_device,
    load_archive,
    load_eval_model,
    make_novelty_archive,
    resolve_autocast,
    save_archive,
    save_json,
    write_corpus_zarr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate novelty-biased MetaOthello self-play data.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output Zarr path.")
    parser.add_argument("--game", "--game_alias", dest="game", type=str, default="classic")
    parser.add_argument("--num_games", type=int, default=10_000)
    parser.add_argument("--novelty", type=str, choices=["knn", "count"], default="knn")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=32, help="Only used for kNN novelty.")
    parser.add_argument("--archive_size", type=int, default=50_000)
    parser.add_argument("--archive_in", "--load_archive", dest="archive_in", type=Path, default=None)
    parser.add_argument("--archive_out", "--save_archive", dest="archive_out", type=Path, default=None)
    parser.add_argument("--stats_out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--deterministic_first_move", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args.seed)

    device = get_device()
    autocast = resolve_autocast(device, amp_dtype=args.amp_dtype, enabled=args.amp_dtype != "none")
    tok = Tokenizer()
    model = load_eval_model(args.ckpt, device)

    if args.archive_in is not None:
        archive = load_archive(args.archive_in, args.novelty, args.archive_size)
    else:
        archive = make_novelty_archive(args.novelty, args.archive_size)

    seqs, boards, stats = generate_curious_corpus(
        model,
        tok,
        archive,
        num_games=args.num_games,
        game_alias=args.game,
        beta=args.beta,
        tau=args.tau,
        k=args.k,
        seed_first_move_random=not args.deterministic_first_move,
        autocast=autocast,
        progress=not args.no_progress,
    )

    write_corpus_zarr(seqs, boards, args.out)

    if args.archive_out is not None:
        save_archive(archive, args.archive_out)

    payload = {
        "ckpt": str(args.ckpt),
        "out": str(args.out),
        "game": args.game,
        "novelty": args.novelty,
        "beta": args.beta,
        "tau": args.tau,
        "k": args.k,
        "archive_size": args.archive_size,
        "seed": args.seed,
        "stats": stats,
    }
    if args.stats_out is not None:
        save_json(args.stats_out, payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
