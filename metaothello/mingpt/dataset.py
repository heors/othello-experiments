from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from metaothello.constants import MAX_STEPS
from metaothello.mingpt.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping tokenized Othello move sequences."""

    def __init__(self, data: Any, tokenizer: Tokenizer, tokenize: bool = True) -> None:
        """Initialize the dataset.

        Args:
            data: Iterable of move sequences. Each sequence must be either a plain
                Python list of move strings/None, or an array-like object with a
                `.values` attribute (e.g. xarray DataArray from Zarr).
            tokenizer: Tokenizer used to encode move sequences to integer token IDs.
            tokenize: If True (default), tokenize and pad ``data`` at construction
                time. If False, ``data`` must already be pre-tokenized integer lists
                padded to ``MAX_STEPS``.
        """
        vocab_size = tokenizer.vocab_size
        self.max_len = MAX_STEPS
        self.block_size = self.max_len - 1
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        if tokenize:
            logger.info("Tokenizing data...")
            self.data = self._tokenize(data)
        else:
            self.data = data

        logger.info("Dataset created: %d sequences, %d unique tokens.", len(self.data), vocab_size)

    def _tokenize(self, data: Any) -> list[list[int]]:
        """Tokenize and pad each sequence to max_len.

        Args:
            data: Iterable of move sequences. Each sequence must be either a
                plain Python list of move strings/None, or an array-like object
                with a `.values` attribute (e.g. xarray DataArray, pandas Series)
                that yields move strings/None when iterated.

        Returns:
            List of token-ID lists, each padded to ``self.max_len``.
        """
        t_data = []
        for seq in tqdm(data, desc="Tokenizing"):
            moves: list = seq.values.tolist() if hasattr(seq, "values") else list(seq)
            tokens = self.tokenizer.encode(moves)
            if len(tokens) < self.max_len:
                tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            t_data.append(tokens)

        return t_data

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (x, y) autoregressive pair for the sequence at ``idx``."""
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
