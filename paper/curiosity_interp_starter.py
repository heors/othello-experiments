from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import TensorDataset

from metaothello.analysis_utils import BLOCK_SIZE, VOCAB_SIZE, gen_games, get_device
from metaothello.constants import BOARD_DIM, MAX_STEPS
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.board_probe import LinearProbe, ProbeTrainer
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import load_model_from_ckpt, set_seed

LOGGER = logging.getLogger("curiosity_interp_starter")
PAD_TOKEN_ID = 0
N_LAYERS = 8
MAX_RETRIES = 1000


@dataclass
class EvalBatch:
    seqs: np.ndarray
    boards_abs: np.ndarray
    valid_pos: np.ndarray


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_zarr_seqs(zarr_path: Path, num_games: int | None, seed: int) -> np.ndarray:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("xarray with zarr support is required for --zarr") from exc

    ds = xr.open_zarr(zarr_path)
    n_total = int(ds.sizes["game"])
    if num_games is None or num_games <= 0 or num_games >= n_total:
        return ds["seqs"].values.astype(np.int32)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=num_games, replace=False)).astype(np.int64)
    return ds["seqs"].isel(game=idx).values.astype(np.int32)


def _pad_board_history(board_history: np.ndarray, total_len: int) -> np.ndarray:
    if board_history.shape[0] >= total_len:
        return board_history[:total_len]
    if board_history.shape[0] == 0:
        pad = np.zeros((total_len, BOARD_DIM, BOARD_DIM), dtype=np.float32)
        return pad
    last = board_history[-1:]
    reps = np.repeat(last, total_len - board_history.shape[0], axis=0)
    return np.concatenate([board_history, reps], axis=0)


def replay_boards_from_seqs(
    seqs: np.ndarray,
    *,
    tok: Tokenizer,
    game_alias: str,
) -> np.ndarray:
    """Reconstruct board histories by replaying token sequences with the game engine.

    This is slower than loading cached board states but avoids the huge activation caches used
    in the official repo. It is intended for smaller exploratory runs.
    """
    game_cls = GAME_REGISTRY[game_alias]
    boards_all: list[np.ndarray] = []

    for seq in seqs:
        game = game_cls()
        moves = tok.decode(seq.tolist())
        # Stop at PAD; PASS (None) is a valid move and should be replayed.
        for move in moves:
            if move == tok.PAD_TOKEN:
                break
            game.play_move(move)
        board_history = np.array(game.get_board_history(), dtype=np.float32)
        boards_all.append(_pad_board_history(board_history, seq.shape[0]))

    return np.stack(boards_all, axis=0)


def valid_positions_from_seqs(seqs: np.ndarray, pad_token_id: int = PAD_TOKEN_ID) -> np.ndarray:
    """Return valid teacher-forcing positions of shape (N, T=59).

    Position t is valid if the next token (the target at t+1) is not PAD.
    """
    return seqs[:, 1:] != pad_token_id


def apply_turn_mask(board_states: np.ndarray) -> np.ndarray:
    """Map absolute {-1,0,1} board states to current-player-relative states.

    Following the later Othello probe setup, even timesteps are multiplied by -1 so the labels
    correspond to Opponent / Empty / Mine relative to the player whose turn is next.
    """
    t = board_states.shape[1]
    turn_mask = np.ones(t, dtype=np.float32)
    turn_mask[::2] = -1.0
    return board_states * turn_mask.reshape(1, -1, 1, 1)


def load_eval_batch(
    *,
    game_alias: str,
    num_games: int,
    seed: int,
    zarr_path: Path | None,
    need_boards: bool = True,
) -> EvalBatch:
    tok = Tokenizer()
    if zarr_path is None:
        seqs, _valid_masks = gen_games(game_alias, num_games=num_games, tokenizer=tok)
    else:
        seqs = _load_zarr_seqs(zarr_path, num_games=num_games, seed=seed)
    boards_abs = replay_boards_from_seqs(seqs, tok=tok, game_alias=game_alias) if need_boards else np.empty((0,), dtype=np.float32)
    valid_pos = valid_positions_from_seqs(seqs)
    return EvalBatch(seqs=seqs, boards_abs=boards_abs, valid_pos=valid_pos)


def _resid_post_filter(name: str) -> bool:
    return "hook_resid_post" in name


@torch.inference_mode()
def extract_logits_and_acts(
    ckpt_path: Path,
    *,
    seqs: np.ndarray,
    layers: Iterable[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    model = load_model_from_ckpt(ckpt_path, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    layers = sorted(set(int(x) for x in layers))
    logits_batches: list[np.ndarray] = []
    layer_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for start in range(0, len(seqs), batch_size):
        end = min(start + batch_size, len(seqs))
        x = torch.tensor(seqs[start:end, :-1], dtype=torch.long, device=device)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits, cache = model.run_with_cache(x, names_filter=_resid_post_filter)
        logits_batches.append(logits.float().cpu().numpy())

        stacked = cache.stack_activation("resid_post").permute(1, 2, 3, 0).float().cpu().numpy()
        for layer in layers:
            layer_batches[layer].append(stacked[:, :, :, layer - 1])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_logits = np.concatenate(logits_batches, axis=0)
    all_layers = {layer: np.concatenate(chunks, axis=0) for layer, chunks in layer_batches.items()}
    return all_logits, all_layers


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def js_divergence_by_move(logits_a: np.ndarray, logits_b: np.ndarray, valid_pos: np.ndarray) -> list[float]:
    p = softmax_np(logits_a)
    q = softmax_np(logits_b)
    m = 0.5 * (p + q)
    eps = 1e-12
    js = 0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)), axis=-1)
    js += 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)), axis=-1)

    out: list[float] = []
    for t in range(js.shape[1]):
        mask_t = valid_pos[:, t]
        if mask_t.any():
            out.append(float(js[mask_t, t].mean()))
        else:
            out.append(float("nan"))
    return out


def top1_disagreement_by_move(logits_a: np.ndarray, logits_b: np.ndarray, valid_pos: np.ndarray) -> list[float]:
    top_a = logits_a.argmax(axis=-1)
    top_b = logits_b.argmax(axis=-1)
    disagree = (top_a != top_b).astype(np.float32)

    out: list[float] = []
    for t in range(disagree.shape[1]):
        mask_t = valid_pos[:, t]
        if mask_t.any():
            out.append(float(disagree[mask_t, t].mean()))
        else:
            out.append(float("nan"))
    return out


def activation_cosine(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    valid_pos: np.ndarray,
) -> tuple[list[float], float]:
    """Cosine similarity between mean activations at each move position."""
    move_values: list[float] = []
    for t in range(acts_a.shape[1]):
        mask_t = valid_pos[:, t]
        if not mask_t.any():
            move_values.append(float("nan"))
            continue
        mean_a = acts_a[mask_t, t].mean(axis=0)
        mean_b = acts_b[mask_t, t].mean(axis=0)
        denom = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
        val = float(np.dot(mean_a, mean_b) / max(denom, 1e-12))
        move_values.append(val)
    mean_val = float(np.nanmean(np.asarray(move_values, dtype=np.float64)))
    return move_values, mean_val


def _split_games(
    seqs: np.ndarray,
    boards_rel: np.ndarray,
    valid_pos: np.ndarray,
    *,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(seqs))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - test_frac))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return (
        seqs[train_idx],
        boards_rel[train_idx],
        valid_pos[train_idx],
        seqs[test_idx],
        boards_rel[test_idx],
        valid_pos[test_idx],
    )


def flatten_probe_examples(
    acts: np.ndarray,
    boards_rel: np.ndarray,
    valid_pos: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, t, d = acts.shape
    x = acts.reshape(n * t, d)
    y = boards_rel[:, :t].reshape(n * t, BOARD_DIM * BOARD_DIM)
    m = valid_pos.reshape(n * t)
    x = x[m]
    y = y[m]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


def evaluate_probe_metrics(
    probe: LinearProbe,
    acts: np.ndarray,
    boards_rel: np.ndarray,
    valid_pos: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    n, t, d = acts.shape
    acts_flat = acts.reshape(n * t, d)
    boards_flat = boards_rel[:, :t].reshape(n * t, BOARD_DIM * BOARD_DIM).astype(np.int64)

    preds_all: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n * t, batch_size):
            end = min(start + batch_size, n * t)
            x = torch.tensor(acts_flat[start:end], dtype=torch.float32, device=device)
            logits, _ = probe(x)
            pred = logits.argmax(dim=-1).cpu().numpy() - 1
            preds_all.append(pred)

    preds = np.concatenate(preds_all, axis=0).reshape(n, t, BOARD_DIM * BOARD_DIM)
    trues = boards_rel[:, :t].reshape(n, t, BOARD_DIM * BOARD_DIM).astype(np.int64)

    cell_per_pos = (preds == trues).mean(axis=-1)
    board_per_pos = (preds == trues).all(axis=-1).astype(np.float32)

    summary = {
        "cell_acc_mean": float(cell_per_pos[valid_pos].mean()),
        "board_acc_mean": float(board_per_pos[valid_pos].mean()),
        "num_valid_positions": int(valid_pos.sum()),
    }

    per_move_cell: list[float] = []
    per_move_board: list[float] = []
    for move_idx in range(t):
        mask_t = valid_pos[:, move_idx]
        if mask_t.any():
            per_move_cell.append(float(cell_per_pos[mask_t, move_idx].mean()))
            per_move_board.append(float(board_per_pos[mask_t, move_idx].mean()))
        else:
            per_move_cell.append(float("nan"))
            per_move_board.append(float("nan"))

    summary["cell_acc_by_move"] = per_move_cell
    summary["board_acc_by_move"] = per_move_board
    return summary


def fit_pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def plot_lines(
    xs: np.ndarray,
    ys_by_label: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for label, ys in ys_by_label.items():
        ax.plot(xs, ys, label=label)
    ax.set_title(title)
    ax.set_xlabel("Move number")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pca_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    unique = list(dict.fromkeys(labels.tolist()))
    for label in unique:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.35, label=str(label))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_summary(
    layers: list[int],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.plot(layers, values, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_compare_overview(
    *,
    layers: list[int],
    act_summary: dict[str, object],
    js_by_move: list[float],
    disagree_by_move: list[float],
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    act_means = [float(act_summary[str(layer)]["mean"]) for layer in layers]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), width_ratios=[1.15, 1.0, 1.1])

    for layer in layers:
        axes[0].plot(
            np.arange(1, len(act_summary[str(layer)]["by_move"]) + 1),
            act_summary[str(layer)]["by_move"],
            linewidth=2,
            label=f"L{layer}",
        )
    axes[0].set_title("Activation cosine by move")
    axes[0].set_xlabel("Move number")
    axes[0].set_ylabel("Cosine")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(layers, act_means, marker="o", linewidth=2)
    axes[1].set_title("Mean activation cosine by layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean cosine")
    axes[1].set_ylim(top=1.02)
    axes[1].grid(True, alpha=0.2)

    moves = np.arange(1, len(js_by_move) + 1)
    axes[2].plot(moves, js_by_move, linewidth=2, label="JS divergence")
    axes[2].plot(moves, disagree_by_move, linewidth=2, label="Top-1 disagreement")
    axes[2].set_title(f"Output divergence: {label_a} vs {label_b}")
    axes[2].set_xlabel("Move number")
    axes[2].set_ylabel("Divergence")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _sample_paired_valid_activations(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    valid_pos: np.ndarray,
    *,
    points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_a = acts_a[valid_pos]
    flat_b = acts_b[valid_pos]
    n = min(len(flat_a), len(flat_b), points)
    if n <= 1:
        raise ValueError("Need at least two valid activation points for representational similarity")
    rng = np.random.default_rng(seed)
    idx = rng.choice(min(len(flat_a), len(flat_b)), size=n, replace=False)
    return flat_a[idx], flat_b[idx]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, ord="fro") ** 2
    norm_x = np.linalg.norm(x.T @ x, ord="fro")
    norm_y = np.linalg.norm(y.T @ y, ord="fro")
    return float(hsic / max(norm_x * norm_y, 1e-12))


def rsa_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    sim_x = x @ x.T
    sim_y = y @ y.T
    tri = np.triu_indices(sim_x.shape[0], k=1)
    corr = spearmanr(sim_x[tri], sim_y[tri]).correlation
    return float(corr if corr is not None else float("nan"))


def representational_similarity_by_layer(
    acts_a_map: dict[int, np.ndarray],
    acts_b_map: dict[int, np.ndarray],
    valid_pos: np.ndarray,
    *,
    layers: list[int],
    rep_points: int,
    rsa_points: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for offset, layer in enumerate(layers):
        x_cka, y_cka = _sample_paired_valid_activations(
            acts_a_map[layer],
            acts_b_map[layer],
            valid_pos,
            points=rep_points,
            seed=seed + 17 * offset,
        )
        x_rsa, y_rsa = _sample_paired_valid_activations(
            acts_a_map[layer],
            acts_b_map[layer],
            valid_pos,
            points=min(rep_points, rsa_points),
            seed=seed + 101 + 17 * offset,
        )
        out[str(layer)] = {
            "cka": linear_cka(x_cka, y_cka),
            "cka_points": int(x_cka.shape[0]),
            "rsa_spearman": rsa_spearman(x_rsa, y_rsa),
            "rsa_points": int(x_rsa.shape[0]),
        }
    return out


def plot_rep_similarity(
    layers: list[int],
    rep_summary: dict[str, dict[str, float | int]],
    *,
    out_path: Path,
) -> None:
    cka = [float(rep_summary[str(layer)]["cka"]) for layer in layers]
    rsa = [float(rep_summary[str(layer)]["rsa_spearman"]) for layer in layers]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    axes[0].plot(layers, cka, marker="o", linewidth=2)
    axes[0].set_title("Linear CKA")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Similarity")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(layers, rsa, marker="o", linewidth=2)
    axes[1].set_title("RSA (Spearman)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Similarity")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cross_layer_similarity_matrices(
    acts_a_map: dict[int, np.ndarray],
    acts_b_map: dict[int, np.ndarray],
    valid_pos: np.ndarray,
    *,
    layers: list[int],
    cka_points: int,
    rsa_points: int,
    seed: int,
) -> dict[str, np.ndarray]:
    cka_mat = np.zeros((len(layers), len(layers)), dtype=np.float64)
    rsa_mat = np.zeros((len(layers), len(layers)), dtype=np.float64)

    cka_samples: dict[int, np.ndarray] = {}
    rsa_samples: dict[int, np.ndarray] = {}
    for offset, layer in enumerate(layers):
        cka_x, _ = _sample_paired_valid_activations(
            acts_a_map[layer],
            acts_a_map[layer],
            valid_pos,
            points=cka_points,
            seed=seed + 31 * offset,
        )
        rsa_x, _ = _sample_paired_valid_activations(
            acts_a_map[layer],
            acts_a_map[layer],
            valid_pos,
            points=min(cka_points, rsa_points),
            seed=seed + 211 + 31 * offset,
        )
        cka_samples[layer] = cka_x
        rsa_samples[layer] = rsa_x

    for i, layer_a in enumerate(layers):
        for j, layer_b in enumerate(layers):
            cka_y, _ = _sample_paired_valid_activations(
                acts_b_map[layer_b],
                acts_b_map[layer_b],
                valid_pos,
                points=cka_samples[layer_a].shape[0],
                seed=seed + 1001 + 53 * i + j,
            )
            rsa_y, _ = _sample_paired_valid_activations(
                acts_b_map[layer_b],
                acts_b_map[layer_b],
                valid_pos,
                points=rsa_samples[layer_a].shape[0],
                seed=seed + 2001 + 53 * i + j,
            )
            cka_mat[i, j] = linear_cka(cka_samples[layer_a], cka_y)
            rsa_mat[i, j] = rsa_spearman(rsa_samples[layer_a], rsa_y)

    return {"cka": cka_mat, "rsa_spearman": rsa_mat}


def plot_similarity_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    *,
    title: str,
    out_path: Path,
    value_fmt: str = ".2f",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Curious layer")
    ax.set_ylabel("Fixed layer")
    ax.set_xticks(np.arange(len(layers)), labels=layers)
    ax.set_yticks(np.arange(len(layers)), labels=layers)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(float(matrix[i, j]), value_fmt), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("Similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compare_command(args: argparse.Namespace) -> None:
    device = get_device()
    layers = sorted(set(args.layers))
    batch = load_eval_batch(
        game_alias=args.game,
        num_games=args.num_games,
        seed=args.seed,
        zarr_path=args.zarr,
        need_boards=False,
    )

    logits_a, acts_a = extract_logits_and_acts(
        args.ckpt_a,
        seqs=batch.seqs,
        layers=layers,
        batch_size=args.batch_size,
        device=device,
    )
    logits_b, acts_b = extract_logits_and_acts(
        args.ckpt_b,
        seqs=batch.seqs,
        layers=layers,
        batch_size=args.batch_size,
        device=device,
    )

    js_by_move = js_divergence_by_move(logits_a, logits_b, batch.valid_pos)
    disagree_by_move = top1_disagreement_by_move(logits_a, logits_b, batch.valid_pos)

    act_summary: dict[str, object] = {}
    plot_payload: dict[str, list[float]] = {}
    for layer in layers:
        by_move, mean_val = activation_cosine(acts_a[layer], acts_b[layer], batch.valid_pos)
        act_summary[str(layer)] = {
            "mean": mean_val,
            "by_move": by_move,
        }
        plot_payload[f"L{layer}"] = by_move

    rep_summary = representational_similarity_by_layer(
        acts_a,
        acts_b,
        batch.valid_pos,
        layers=layers,
        rep_points=args.rep_points,
        rsa_points=args.rsa_points,
        seed=args.seed,
    )
    cross_layer = cross_layer_similarity_matrices(
        acts_a,
        acts_b,
        batch.valid_pos,
        layers=layers,
        cka_points=args.rep_points,
        rsa_points=args.rsa_points,
        seed=args.seed,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_lines(
        np.arange(1, BLOCK_SIZE + 1),
        plot_payload,
        title="Activation cosine similarity",
        ylabel="Cosine",
        out_path=out_dir / "activation_cosine_by_move.png",
    )
    plot_layer_summary(
        layers,
        [float(act_summary[str(layer)]["mean"]) for layer in layers],
        title="Mean activation cosine by layer",
        ylabel="Mean cosine",
        out_path=out_dir / "activation_cosine_by_layer.png",
    )
    plot_lines(
        np.arange(1, BLOCK_SIZE + 1),
        {"JS": js_by_move},
        title="Policy divergence",
        ylabel="Jensen-Shannon divergence",
        out_path=out_dir / "policy_js_by_move.png",
    )
    plot_lines(
        np.arange(1, BLOCK_SIZE + 1),
        {"Top-1 disagreement": disagree_by_move},
        title="Top-1 disagreement",
        ylabel="Rate",
        out_path=out_dir / "top1_disagreement_by_move.png",
    )
    plot_compare_overview(
        layers=layers,
        act_summary=act_summary,
        js_by_move=js_by_move,
        disagree_by_move=disagree_by_move,
        label_a=args.label_a,
        label_b=args.label_b,
        out_path=out_dir / "compare_overview.png",
    )
    plot_rep_similarity(
        layers,
        rep_summary,
        out_path=out_dir / "representation_similarity_by_layer.png",
    )
    plot_similarity_heatmap(
        cross_layer["cka"],
        layers,
        title="Cross-layer linear CKA",
        out_path=out_dir / "cross_layer_cka_heatmap.png",
    )
    plot_similarity_heatmap(
        cross_layer["rsa_spearman"],
        layers,
        title="Cross-layer RSA (Spearman)",
        out_path=out_dir / "cross_layer_rsa_heatmap.png",
    )

    if args.pca_layer is not None:
        layer = int(args.pca_layer)
        if layer not in acts_a:
            raise ValueError(f"PCA layer {layer} not present in extracted layers {layers}")
        acts_flat_a = acts_a[layer][batch.valid_pos]
        acts_flat_b = acts_b[layer][batch.valid_pos]
        n_each = min(len(acts_flat_a), len(acts_flat_b), args.pca_points)
        rng = np.random.default_rng(args.seed)
        idx_a = rng.choice(len(acts_flat_a), size=n_each, replace=False)
        idx_b = rng.choice(len(acts_flat_b), size=n_each, replace=False)
        stacked = np.concatenate([acts_flat_a[idx_a], acts_flat_b[idx_b]], axis=0)
        labels = np.array([args.label_a] * n_each + [args.label_b] * n_each)
        coords = fit_pca_2d(stacked)
        plot_pca_scatter(
            coords,
            labels,
            title=f"Layer {layer} activation PCA",
            out_path=out_dir / f"activation_pca_L{layer}.png",
        )

    summary = {
        "game": args.game,
        "num_games": int(batch.seqs.shape[0]),
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "layers": layers,
        "policy": {
            "js_mean": float(np.nanmean(np.asarray(js_by_move))),
            "js_by_move": js_by_move,
            "top1_disagreement_mean": float(np.nanmean(np.asarray(disagree_by_move))),
            "top1_disagreement_by_move": disagree_by_move,
        },
        "activation_cosine": act_summary,
        "representation_similarity": rep_summary,
        "cross_layer_similarity": {
            "layers": layers,
            "cka": cross_layer["cka"].tolist(),
            "rsa_spearman": cross_layer["rsa_spearman"].tolist(),
        },
    }
    save_json(out_dir / "summary.json", summary)
    LOGGER.info("Wrote outputs to %s", out_dir)


def _trainer_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "lr": args.lr,
        "wd": args.wd,
        "betas": [0.9, 0.95],
        "batch_size": args.probe_batch_size,
        "num_workers": 0,
        "grad_norm_clip": 1.0,
        "max_epochs": args.epochs,
    }


def train_probe_command(args: argparse.Namespace) -> None:
    device = get_device()
    batch = load_eval_batch(
        game_alias=args.game,
        num_games=args.num_games,
        seed=args.seed,
        zarr_path=args.zarr,
        need_boards=True,
    )
    boards_rel = apply_turn_mask(batch.boards_abs)

    seqs_tr, boards_tr, valid_tr, seqs_te, boards_te, valid_te = _split_games(
        batch.seqs,
        boards_rel,
        batch.valid_pos,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    _logits_tr, acts_tr_map = extract_logits_and_acts(
        args.ckpt,
        seqs=seqs_tr,
        layers=[args.layer],
        batch_size=args.batch_size,
        device=device,
    )
    _logits_te, acts_te_map = extract_logits_and_acts(
        args.ckpt,
        seqs=seqs_te,
        layers=[args.layer],
        batch_size=args.batch_size,
        device=device,
    )
    acts_tr = acts_tr_map[args.layer]
    acts_te = acts_te_map[args.layer]

    x_train, y_train = flatten_probe_examples(acts_tr, boards_tr, valid_tr)
    x_test, y_test = flatten_probe_examples(acts_te, boards_te, valid_te)

    probe = LinearProbe(device=torch.cuda.current_device() if torch.cuda.is_available() else device)
    trainer = ProbeTrainer(
        probe,
        TensorDataset(x_train, y_train),
        TensorDataset(x_test, y_test),
        _trainer_config(args),
    )
    trainer.train()

    out_ckpt = args.out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), out_ckpt)

    metrics = evaluate_probe_metrics(
        probe,
        acts_te,
        boards_te,
        valid_te,
        batch_size=args.probe_batch_size,
        device=device,
    )
    metrics.update(
        {
            "ckpt": str(args.ckpt),
            "game": args.game,
            "layer": int(args.layer),
            "num_games": int(batch.seqs.shape[0]),
            "num_train_examples": int(x_train.shape[0]),
            "num_test_examples": int(x_test.shape[0]),
            "probe_ckpt": str(out_ckpt),
        }
    )
    save_json(args.metrics_out, metrics)
    LOGGER.info("Probe saved to %s", out_ckpt)
    LOGGER.info("Metrics saved to %s", args.metrics_out)


def load_probe_ckpt(probe_path: Path, device: torch.device) -> LinearProbe:
    probe = LinearProbe(device=device)
    state = torch.load(probe_path, map_location=device)
    probe.load_state_dict(state)
    probe.eval()
    return probe


def eval_probe_command(args: argparse.Namespace) -> None:
    device = get_device()
    batch = load_eval_batch(
        game_alias=args.game,
        num_games=args.num_games,
        seed=args.seed,
        zarr_path=args.zarr,
        need_boards=True,
    )
    boards_rel = apply_turn_mask(batch.boards_abs)

    _logits, acts_map = extract_logits_and_acts(
        args.ckpt,
        seqs=batch.seqs,
        layers=[args.layer],
        batch_size=args.batch_size,
        device=device,
    )
    probe = load_probe_ckpt(args.probe, device)
    metrics = evaluate_probe_metrics(
        probe,
        acts_map[args.layer],
        boards_rel,
        batch.valid_pos,
        batch_size=args.probe_batch_size,
        device=device,
    )
    metrics.update(
        {
            "probe": str(args.probe),
            "ckpt": str(args.ckpt),
            "game": args.game,
            "layer": int(args.layer),
            "num_games": int(batch.seqs.shape[0]),
        }
    )
    save_json(args.metrics_out, metrics)
    LOGGER.info("Metrics saved to %s", args.metrics_out)


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(denom, 1e-12)


def _load_probe_weight(probe_path: Path) -> np.ndarray:
    state = torch.load(probe_path, map_location="cpu")
    weight = state.get("proj.weight")
    if weight is None:
        raise KeyError(f"{probe_path} does not contain proj.weight")
    arr = weight.detach().cpu().numpy().astype(np.float64)
    if arr.shape[0] != BOARD_DIM * BOARD_DIM * 3:
        raise ValueError(f"Unexpected probe weight shape {arr.shape} for {probe_path}")
    return arr


def probe_geometry_command(args: argparse.Namespace) -> None:
    a = _load_probe_weight(args.probe_a)
    b = _load_probe_weight(args.probe_b)

    raw = _rowwise_cosine(a, b)

    # Orthogonal Procrustes: find R minimizing ||B R - A||_F.
    u, _s, vt = np.linalg.svd(b.T @ a, full_matrices=False)
    r = u @ vt
    b_aligned = b @ r
    aligned = _rowwise_cosine(a, b_aligned)

    summary = {
        "probe_a": str(args.probe_a),
        "probe_b": str(args.probe_b),
        "raw_cosine_mean": float(raw.mean()),
        "raw_cosine_std": float(raw.std(ddof=1)),
        "aligned_cosine_mean": float(aligned.mean()),
        "aligned_cosine_std": float(aligned.std(ddof=1)),
        "raw_cosine_per_row": raw.tolist(),
        "aligned_cosine_per_row": aligned.tolist(),
    }
    save_json(args.metrics_out, summary)

    if args.plot_pca is not None:
        stacked = np.vstack([a, b])
        coords = fit_pca_2d(stacked)
        labels = np.array([args.label_a] * len(a) + [args.label_b] * len(b))
        plot_pca_scatter(coords, labels, title="Probe weight PCA", out_path=args.plot_pca)

    LOGGER.info("Metrics saved to %s", args.metrics_out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Starter interpretability utilities for comparing Othello checkpoints.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    sub = p.add_subparsers(dest="cmd", required=True)

    p_cmp = sub.add_parser("compare", help="Compare two checkpoints on shared sequences.")
    p_cmp.add_argument("--ckpt_a", type=Path, required=True)
    p_cmp.add_argument("--ckpt_b", type=Path, required=True)
    p_cmp.add_argument("--label_a", type=str, default="standard")
    p_cmp.add_argument("--label_b", type=str, default="curious")
    p_cmp.add_argument("--game", type=str, default="classic")
    p_cmp.add_argument("--num_games", type=int, default=1000)
    p_cmp.add_argument("--batch_size", type=int, default=128)
    p_cmp.add_argument("--layers", type=int, nargs="+", default=[1, 5, 8])
    p_cmp.add_argument("--zarr", type=Path, default=None)
    p_cmp.add_argument("--out_dir", type=Path, required=True)
    p_cmp.add_argument("--pca_layer", type=int, default=None)
    p_cmp.add_argument("--pca_points", type=int, default=5000)
    p_cmp.add_argument("--rep_points", type=int, default=1024, help="Number of valid positions for layerwise CKA.")
    p_cmp.add_argument("--rsa_points", type=int, default=512, help="Subset size for RSA similarity matrices.")
    p_cmp.set_defaults(func=compare_command)

    p_tr = sub.add_parser("train-probe", help="Train a small linear board probe on one checkpoint/layer.")
    p_tr.add_argument("--ckpt", type=Path, required=True)
    p_tr.add_argument("--game", type=str, default="classic")
    p_tr.add_argument("--layer", type=int, required=True)
    p_tr.add_argument("--num_games", type=int, default=2000)
    p_tr.add_argument("--batch_size", type=int, default=128, help="Model forward batch size.")
    p_tr.add_argument("--probe_batch_size", type=int, default=1024)
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--wd", type=float, default=0.0)
    p_tr.add_argument("--test_frac", type=float, default=0.2)
    p_tr.add_argument("--zarr", type=Path, default=None)
    p_tr.add_argument("--out_ckpt", type=Path, required=True)
    p_tr.add_argument("--metrics_out", type=Path, required=True)
    p_tr.set_defaults(func=train_probe_command)

    p_ev = sub.add_parser("eval-probe", help="Evaluate a probe on a checkpoint/layer.")
    p_ev.add_argument("--probe", type=Path, required=True)
    p_ev.add_argument("--ckpt", type=Path, required=True)
    p_ev.add_argument("--game", type=str, default="classic")
    p_ev.add_argument("--layer", type=int, required=True)
    p_ev.add_argument("--num_games", type=int, default=1000)
    p_ev.add_argument("--batch_size", type=int, default=128)
    p_ev.add_argument("--probe_batch_size", type=int, default=1024)
    p_ev.add_argument("--zarr", type=Path, default=None)
    p_ev.add_argument("--metrics_out", type=Path, required=True)
    p_ev.set_defaults(func=eval_probe_command)

    p_geom = sub.add_parser("probe-geometry", help="Compare two probe checkpoints in weight space.")
    p_geom.add_argument("--probe_a", type=Path, required=True)
    p_geom.add_argument("--probe_b", type=Path, required=True)
    p_geom.add_argument("--label_a", type=str, default="A")
    p_geom.add_argument("--label_b", type=str, default="B")
    p_geom.add_argument("--metrics_out", type=Path, required=True)
    p_geom.add_argument("--plot_pca", type=Path, default=None)
    p_geom.set_defaults(func=probe_geometry_command)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
