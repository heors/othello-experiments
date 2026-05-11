"""Small support utilities used by the continuation experiment helpers."""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm

from metaothello.constants import MAX_STEPS
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer

LOGGER = logging.getLogger(__name__)

VOCAB_SIZE = 66
BLOCK_SIZE = MAX_STEPS - 1
_MAX_RETRIES = 1000


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, "mps") and torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)
    return device


def gen_games(
    game_alias: str,
    num_games: int,
    tokenizer: Tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate complete random games and per-step valid-move masks."""
    game_class = GAME_REGISTRY[game_alias]
    seqs: list[list[int]] = []
    valid_masks: list[np.ndarray] = []
    vocab_size = tokenizer.vocab_size

    for _ in tqdm(range(num_games), desc=f"Generating {game_alias} games", leave=False):
        for _attempt in range(_MAX_RETRIES):
            game = game_class()  # type: ignore[operator]
            game.generate_random_game()
            history = game.get_history()
            if len(history) != MAX_STEPS:
                continue

            seqs.append(tokenizer.encode(history))

            replay = game_class()  # type: ignore[operator]
            has_mapping = hasattr(replay, "mapping")
            game_masks = np.zeros((MAX_STEPS, vocab_size), dtype=bool)

            for step in range(MAX_STEPS):
                valid_physical = replay.get_all_valid_moves()
                valid_names = (
                    [replay.mapping[move] for move in valid_physical]
                    if has_mapping
                    else valid_physical
                )
                for name in valid_names:
                    game_masks[step, tokenizer.stoi[name]] = True

                if has_mapping:
                    replay.play_move(replay.reverse_mapping[history[step]])
                else:
                    replay.play_move(history[step])

            valid_masks.append(game_masks)
            break
        else:
            raise RuntimeError(
                f"Could not generate a valid {MAX_STEPS}-step {game_alias} game "
                f"after {_MAX_RETRIES} attempts"
            )

    return np.asarray(seqs, dtype=np.int32), np.asarray(valid_masks, dtype=bool)
