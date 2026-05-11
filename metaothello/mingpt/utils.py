from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
import xarray as xr
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .model import GPT, GPTConfig


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value applied to random, numpy, torch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fresh_model(
    vocab_size: int,
    block_size: int,
    n_layer: int = 8,
    n_head: int = 8,
    n_embd: int = 512,
) -> GPT:
    """Instantiate a GPT model with the given architecture parameters.

    Args:
        vocab_size: Number of token types in the vocabulary.
        block_size: Context window size (sequence length).
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads per block.
        n_embd: Embedding dimension.

    Returns:
        Freshly initialised GPT model on CPU.
    """
    mconf = GPTConfig(vocab_size, block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    return GPT(mconf)


def convert_to_transformer_lens_format(
    in_sd: dict,
    n_layers: int = 8,
    n_heads: int = 8,
) -> dict:
    """Remap a minGPT state dict to the TransformerLens naming convention.

    Args:
        in_sd: State dict from a GPT model.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.

    Returns:
        State dict with keys remapped to TransformerLens format.
    """
    out_sd = {}
    out_sd["pos_embed.W_pos"] = in_sd["pos_emb"].squeeze(0)
    out_sd["embed.W_E"] = in_sd["tok_emb.weight"]

    out_sd["ln_final.w"] = in_sd["ln_f.weight"]
    out_sd["ln_final.b"] = in_sd["ln_f.bias"]
    out_sd["unembed.W_U"] = in_sd["head.weight"].T

    for layer in range(n_layers):
        out_sd[f"blocks.{layer}.ln1.w"] = in_sd[f"blocks.{layer}.ln1.weight"]
        out_sd[f"blocks.{layer}.ln1.b"] = in_sd[f"blocks.{layer}.ln1.bias"]
        out_sd[f"blocks.{layer}.ln2.w"] = in_sd[f"blocks.{layer}.ln2.weight"]
        out_sd[f"blocks.{layer}.ln2.b"] = in_sd[f"blocks.{layer}.ln2.bias"]

        out_sd[f"blocks.{layer}.attn.W_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_O"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.proj.weight"],
            "d_model (head d_head) -> head d_head d_model",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_O"] = in_sd[f"blocks.{layer}.attn.proj.bias"]

        out_sd[f"blocks.{layer}.mlp.b_in"] = in_sd[f"blocks.{layer}.mlp.0.bias"]
        out_sd[f"blocks.{layer}.mlp.W_in"] = in_sd[f"blocks.{layer}.mlp.0.weight"].T
        out_sd[f"blocks.{layer}.mlp.b_out"] = in_sd[f"blocks.{layer}.mlp.2.bias"]
        out_sd[f"blocks.{layer}.mlp.W_out"] = in_sd[f"blocks.{layer}.mlp.2.weight"].T

    return out_sd


def load_model_from_ckpt(
    ckpt_path: Path,
    vocab_size: int,
    block_size: int,
    as_tlens: bool = False,
) -> GPT | HookedTransformer:
    """Load a GPT model from a checkpoint file.

    Args:
        ckpt_path: Path to the .ckpt file saved by Trainer.
        vocab_size: Number of token types in the vocabulary.
        block_size: Context window size (sequence length).
        as_tlens: If True, return a HookedTransformer (TransformerLens) model.
            If False (default), return the base GPT model.

    Returns:
        Loaded model (GPT or HookedTransformer depending on as_tlens).
    """
    model_base = load_fresh_model(vocab_size, block_size)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model_base = model_base.to(device)
    elif torch.mps.is_available():
        device = torch.device("mps")
        model_base = model_base.to(device)

    if torch.cuda.is_available():
        model_base.load_state_dict(torch.load(ckpt_path))
    else:
        model_base.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))

    if not as_tlens:
        return model_base

    mcfg = GPTConfig(vocab_size, block_size, n_layer=8, n_head=8, n_embd=512)
    cfg = HookedTransformerConfig(
        n_layers=mcfg.n_layer,
        d_model=mcfg.n_embd,
        d_head=mcfg.n_embd // mcfg.n_head,
        n_heads=mcfg.n_head,
        d_mlp=mcfg.n_embd * 4,
        d_vocab=mcfg.vocab_size,
        n_ctx=mcfg.block_size,
        act_fn="gelu",
        normalization_type="LNPre",
    )

    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(convert_to_transformer_lens_format(model_base.state_dict()))

    return model


def shuffle_data(data: Any) -> Any:
    """Return a shuffled copy of data without duplicating the full array.

    Args:
        data: Array-like or xarray DataArray indexed along the first dimension.

    Returns:
        Data with rows permuted in a random order.
    """
    n = len(data)
    data_idx = np.arange(n)
    np.random.shuffle(data_idx)

    return data[data_idx]


def split_train_test(data: Any, test_frac: float = 0.1) -> tuple[Any, Any]:
    """Split data into train and test subsets.

    Args:
        data: Array-like or xarray DataArray with a leading 'game' dimension.
        test_frac: Fraction of data to reserve for the test split.

    Returns:
        Tuple of (train_data, test_data).
    """
    n = len(data)
    train_idx = np.random.choice(range(n), size=int(n * (1 - test_frac)), replace=False)
    test_idx = list(set(range(n)) - set(train_idx))
    if hasattr(data, "isel"):
        return data.isel(game=train_idx), data.isel(game=test_idx)
    return data[train_idx], data[test_idx]


def get_dataset(data_path: Path) -> np.ndarray:
    """Open a Zarr game dataset and return the pre-tokenised sequence array.

    Args:
        data_path: Path to the .zarr store produced by generate_data.py.

    Returns:
        Numpy array of shape (num_games, MAX_STEPS) with dtype int32.
    """
    ds = xr.open_zarr(data_path)
    return ds["seqs"].values


def get_last_ckpt(ckpt_dir: Path) -> tuple[Path | None, int]:
    """Return the most recent checkpoint file and its epoch number.

    Args:
        ckpt_dir: Directory containing .ckpt files named epoch_{n}.ckpt.

    Returns:
        Tuple of (checkpoint_path, epoch_number). Returns (None, 0) if the
        directory does not exist or contains no checkpoints.
    """
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
        return None, 0
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None, 0
    ckpts = sorted(ckpts, key=lambda x: int(x.stem.split("_")[-1]))
    last_ckpt = ckpts[-1]
    last_epoch = int(last_ckpt.stem.split("_")[-1])
    return last_ckpt, last_epoch
