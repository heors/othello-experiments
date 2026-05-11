from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BOARD_DIM = 8
TILE_STATES = 3
EPS = 1e-12

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "train_loss": ("train_loss", "mean_train_loss"),
    "val_loss": ("val_loss",),
    "eval_top1": ("eval_top1",),
    "eval_correct_prob": ("eval_correct_prob",),
    "eval_nll": ("eval_nll",),
    "eval_perplexity": ("eval_perplexity",),
    "gen_novelty": ("mean_gen_novelty", "gen_novelty_mean"),
    "lr": ("lr",),
    "tokens_seen": ("tokens_seen",),
}

METRIC_LABELS: dict[str, str] = {
    "train_loss": "Train loss",
    "val_loss": "Validation loss",
    "eval_top1": "Eval top-1",
    "eval_correct_prob": "Eval correct-token prob.",
    "eval_nll": "Eval NLL",
    "eval_perplexity": "Eval perplexity",
    "gen_novelty": "Generated novelty",
    "lr": "Learning rate",
    "tokens_seen": "Tokens seen",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch_(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Could not parse epoch number from {path}")
    return int(m.group(1))


def load_epoch_payloads(run_dir: Path) -> list[dict]:
    epoch_dir = run_dir / "epoch_logs"
    files = sorted(epoch_dir.glob("epoch_*.json"), key=_epoch_from_path)
    if not files:
        raise FileNotFoundError(f"No epoch log JSON files found in {epoch_dir}")
    return [load_json(path) for path in files]


def metric_value(payload: dict, metric: str) -> float:
    for key in METRIC_ALIASES.get(metric, (metric,)):
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return float("nan")


def _x_values(payloads: list[dict], epoch_origin: int | None) -> list[float]:
    epochs = [float(p["epoch"]) for p in payloads]
    if epoch_origin is None:
        return epochs
    return [epoch - float(epoch_origin) for epoch in epochs]


def _best_payload(payloads: list[dict], metric: str) -> dict | None:
    scored = [(metric_value(payload, metric), payload) for payload in payloads]
    valid = [(value, payload) for value, payload in scored if np.isfinite(value)]
    if not valid:
        return None
    if metric in {"val_loss", "train_loss", "eval_nll", "eval_perplexity"}:
        return min(valid, key=lambda pair: pair[0])[1]
    return max(valid, key=lambda pair: pair[0])[1]


def plot_training_curves(
    histories: dict[str, list[dict]],
    *,
    metrics: list[str],
    out_path: Path,
    title: str,
    epoch_origin: int | None,
) -> None:
    shown_metrics = [
        metric
        for metric in metrics
        if any(np.isfinite(metric_value(payload, metric)) for rows in histories.values() for payload in rows)
    ]
    if not shown_metrics:
        raise ValueError("None of the requested metrics were present in the provided histories")

    n_panels = len(shown_metrics)
    n_cols = min(2, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 3.7 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric in zip(axes_flat, shown_metrics):
        for label, payloads in histories.items():
            xs = _x_values(payloads, epoch_origin)
            ys = [metric_value(payload, metric) for payload in payloads]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Continuation epoch" if epoch_origin is not None else "Epoch")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.2)
        if metric == "eval_top1":
            ax.set_ylim(bottom=0.0)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _strip_prefix(name: str) -> str:
    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
        "net.",
    )
    out = name
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def _extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        tensor_like = all(isinstance(v, torch.Tensor) for v in obj.values()) if obj else False
        if tensor_like:
            return {str(k): v for k, v in obj.items()}
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "weights",
        ):
            if key in obj and isinstance(obj[key], dict):
                maybe = obj[key]
                if maybe and all(isinstance(v, torch.Tensor) for v in maybe.values()):
                    return {str(k): v for k, v in maybe.items()}
    raise ValueError("Could not find a tensor state dict inside checkpoint")


def load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(obj)
    return {_strip_prefix(k): v.detach().cpu() for k, v in state.items()}


_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)blocks\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
    re.compile(r"(?:^|\.)layers\.(\d+)\."),
    re.compile(r"(?:^|\.)transformer\.h\.(\d+)\."),
    re.compile(r"(?:^|\.)transformer\.blocks\.(\d+)\."),
]


def find_layer_idx(name: str) -> int | None:
    for pat in _LAYER_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


_EMBED_HINTS = ("wte", "wpe", "tok_emb", "token_embedding", "embed", "pos_emb")
_UNEMBED_HINTS = ("lm_head", "unembed", "head.weight", "head.bias")
_ATTN_HINTS = ("attn", "attention", "self_attn")
_MLP_HINTS = ("mlp", "ffn", "feedforward", "feed_forward")
_NORM_HINTS = ("ln", "norm")


def bucket_name(param_name: str) -> str:
    lower = param_name.lower()
    if any(h in lower for h in _UNEMBED_HINTS):
        return "unembed"
    if any(h in lower for h in _EMBED_HINTS):
        return "embed"
    if any(h in lower for h in _ATTN_HINTS):
        return "attn"
    if any(h in lower for h in _MLP_HINTS):
        return "mlp"
    if any(h in lower for h in _NORM_HINTS):
        return "norm"
    return "other"


def canonical_weight_rows(state: dict[str, torch.Tensor]) -> tuple[list[str], dict[str, np.ndarray]]:
    layer_ids = sorted({idx for key in state for idx in [find_layer_idx(key)] if idx is not None})
    max_layer = max(layer_ids) if layer_ids else -1
    rows: list[str] = ["embed"]
    for layer in range(max_layer + 1):
        rows.extend([f"L{layer} attn", f"L{layer} mlp"])
    rows.extend(["unembed", "norm", "other"])
    row_map: dict[str, list[float]] = {row: [0.0, 0.0] for row in rows}  # [delta_sq, base_sq]
    return rows, {k: np.array(v, dtype=np.float64) for k, v in row_map.items()}


def row_label_for_param(name: str) -> str:
    bucket = bucket_name(name)
    layer = find_layer_idx(name)
    if bucket in {"attn", "mlp"} and layer is not None:
        return f"L{layer} {bucket}"
    if bucket == "unembed":
        return "unembed"
    if bucket == "embed":
        return "embed"
    if bucket == "norm":
        return "norm"
    return "other"


def compute_base_relative_drift(
    base_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[list[str], np.ndarray]:
    rows, stats = canonical_weight_rows(base_state)
    shared = set(base_state).intersection(target_state)
    for key in shared:
        a = base_state[key]
        b = target_state[key]
        if a.shape != b.shape or not torch.is_floating_point(a):
            continue
        label = row_label_for_param(key)
        diff = (b - a).double().numpy()
        base = a.double().numpy()
        stats[label][0] += float(np.sum(diff * diff))
        stats[label][1] += float(np.sum(base * base))

    ratios = []
    for row in rows:
        delta = math.sqrt(stats[row][0])
        base = math.sqrt(stats[row][1])
        ratios.append(delta / max(base, EPS))
    return rows, np.asarray(ratios, dtype=np.float64)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    out_path: Path,
    annotate: bool = True,
    value_fmt: str = ".3f",
    cmap: str = "viridis",
    log10_values: bool = False,
    cbar_label: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(matrix, dtype=np.float64)
    plot_arr = np.log10(arr + 1e-12) if log10_values else arr

    height = max(3.0, 0.38 * len(row_labels) + 1.6)
    width = max(3.0, 1.1 * len(col_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(plot_arr, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    if cbar_label:
        cbar.set_label(cbar_label)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, format(arr[i, j], value_fmt), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_weight_drift_figure(summary: dict, out_path: Path) -> None:
    ratios = np.asarray(summary["ratio_matrix"], dtype=np.float64)
    rows = list(summary["rows"])
    cols = list(summary["columns"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(4.0, 0.35 * len(rows) + 1.4)), width_ratios=[1.2, 1.0])

    im = axes[0].imshow(np.log10(ratios + 1e-12), aspect="auto")
    axes[0].set_title("Base-relative weight drift")
    axes[0].set_xticks(np.arange(len(cols)), labels=cols)
    axes[0].set_yticks(np.arange(len(rows)), labels=rows)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label("log10 ||ΔW|| / ||W0||")

    layer_rows = [r for r in rows if r.startswith("L")]
    layers = sorted({int(r.split()[0][1:]) for r in layer_rows})
    totals = {col: [] for col in cols}
    for col_idx, col in enumerate(cols):
        for layer in layers:
            mask = [i for i, r in enumerate(rows) if r in {f"L{layer} attn", f"L{layer} mlp"}]
            val = float(np.sum(ratios[mask, col_idx])) if mask else float("nan")
            totals[col].append(val)
        axes[1].plot(layers, totals[col], marker="o", label=col)
    axes[1].set_title("Layer-total drift")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Sum of base-relative drift")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_signed_weight_difference_figure(summary: dict, out_path: Path) -> None:
    ratios = np.asarray(summary["ratio_matrix"], dtype=np.float64)
    rows = list(summary["rows"])
    cols = list(summary["columns"])
    if ratios.shape[1] != 2:
        raise ValueError("Signed difference figure currently expects exactly two checkpoints")

    diff = ratios[:, 1] - ratios[:, 0]
    layer_rows = [r for r in rows if r.startswith("L")]
    layers = sorted({int(r.split()[0][1:]) for r in layer_rows})
    layer_totals = []
    for layer in layers:
        mask = [i for i, r in enumerate(rows) if r in {f"L{layer} attn", f"L{layer} mlp"}]
        layer_totals.append(float(np.sum(diff[mask])) if mask else float("nan"))

    vmax = float(np.max(np.abs(diff))) if diff.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, max(4.0, 0.34 * len(rows) + 1.2)), width_ratios=[1.0, 1.2])

    im = axes[0].imshow(diff[:, None], aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"Signed drift difference\n{cols[1]} - {cols[0]}")
    axes[0].set_xticks([0], labels=[f"{cols[1]} - {cols[0]}"])
    axes[0].set_yticks(np.arange(len(rows)), labels=rows)
    for i, value in enumerate(diff):
        axes[0].text(0, i, f"{value:+.4f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.82)
    cbar.set_label("Signed change in ||ΔW|| / ||W0||")

    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].plot(layers, layer_totals, marker="o", linewidth=2)
    axes[1].set_title("Layer-total signed difference")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(f"({cols[1]} - {cols[0]}) drift")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def weight_drift_command(args: argparse.Namespace) -> None:
    base_state = load_state_dict(args.base_ckpt)
    states = {
        args.label_a: load_state_dict(args.ckpt_a),
        args.label_b: load_state_dict(args.ckpt_b),
    }
    rows: list[str] | None = None
    ratio_cols: list[np.ndarray] = []
    for label, state in states.items():
        cur_rows, ratios = compute_base_relative_drift(base_state, state)
        if rows is None:
            rows = cur_rows
        elif cur_rows != rows:
            raise ValueError("Row mismatch across checkpoints")
        ratio_cols.append(ratios)
    assert rows is not None

    ratio_matrix = np.stack(ratio_cols, axis=1)
    summary = {
        "base_ckpt": str(args.base_ckpt),
        "columns": [args.label_a, args.label_b],
        "rows": rows,
        "ratio_matrix": ratio_matrix.tolist(),
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
    }
    save_json(args.summary_out, summary)
    plot_weight_drift_figure(summary, args.plot_out)


def training_curves_command(args: argparse.Namespace) -> None:
    histories: dict[str, list[dict]] = {}
    for label, run_dir_str in args.run:
        run_dir = Path(run_dir_str)
        histories[label] = load_epoch_payloads(run_dir)

    metrics = list(args.metrics)
    plot_training_curves(
        histories,
        metrics=metrics,
        out_path=args.plot_out,
        title=args.title,
        epoch_origin=args.epoch_origin,
    )

    best_summary: dict[str, dict[str, dict[str, float | int] | None]] = {}
    for label, payloads in histories.items():
        best_summary[label] = {}
        for metric in metrics:
            best = _best_payload(payloads, metric)
            if best is None:
                best_summary[label][metric] = None
                continue
            best_summary[label][metric] = {
                "epoch": int(best["epoch"]),
                "value": metric_value(best, metric),
            }

    summary = {
        "plot_title": args.title,
        "epoch_origin": args.epoch_origin,
        "metrics": metrics,
        "runs": {
            label: {
                "run_dir": run_dir,
                "epochs": [int(payload["epoch"]) for payload in histories[label]],
            }
            for label, run_dir in args.run
        },
        "best": best_summary,
    }
    save_json(args.summary_out, summary)


def weight_drift_difference_command(args: argparse.Namespace) -> None:
    summary = load_json(args.weight_drift_summary)
    plot_signed_weight_difference_figure(summary, args.plot_out)


def _load_probe_weight(probe_path: Path) -> np.ndarray:
    state = torch.load(probe_path, map_location="cpu")
    if isinstance(state, dict):
        weight = None
        if "proj.weight" in state:
            weight = state["proj.weight"]
        else:
            for key, value in state.items():
                if str(key).endswith("proj.weight") and isinstance(value, torch.Tensor):
                    weight = value
                    break
        if isinstance(weight, torch.Tensor):
            arr = weight.detach().cpu().numpy().astype(np.float64)
            if arr.shape[0] == BOARD_DIM * BOARD_DIM * TILE_STATES:
                return arr
    raise ValueError(f"Could not find probe weights in {probe_path}")


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(denom, EPS)


def procrustes_align(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(other.T @ reference, full_matrices=False)
    r = u @ vt
    return other @ r


def tilewise_mean(values: np.ndarray) -> np.ndarray:
    arr = values.reshape(BOARD_DIM * BOARD_DIM, TILE_STATES)
    return arr.mean(axis=1).reshape(BOARD_DIM, BOARD_DIM)


def _metric_path(probe_dir: Path, src: str, dst: str, layer: int) -> Path:
    return probe_dir / f"{src}_on_{dst}_L{layer}.json"


def _probe_path(probe_dir: Path, label: str, layer: int) -> Path:
    return probe_dir / f"{label}_L{layer}.ckpt"


def _load_transfer_matrix(probe_dir: Path, label_a: str, label_b: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    payloads = {
        (label_a, label_a): load_json(_metric_path(probe_dir, label_a, label_a, layer)),
        (label_a, label_b): load_json(_metric_path(probe_dir, label_a, label_b, layer)),
        (label_b, label_a): load_json(_metric_path(probe_dir, label_b, label_a, layer)),
        (label_b, label_b): load_json(_metric_path(probe_dir, label_b, label_b, layer)),
    }
    cell = np.array(
        [
            [payloads[(label_a, label_a)]["cell_acc_mean"], payloads[(label_a, label_b)]["cell_acc_mean"]],
            [payloads[(label_b, label_a)]["cell_acc_mean"], payloads[(label_b, label_b)]["cell_acc_mean"]],
        ],
        dtype=np.float64,
    )
    board = np.array(
        [
            [payloads[(label_a, label_a)]["board_acc_mean"], payloads[(label_a, label_b)]["board_acc_mean"]],
            [payloads[(label_b, label_a)]["board_acc_mean"], payloads[(label_b, label_b)]["board_acc_mean"]],
        ],
        dtype=np.float64,
    )
    return cell, board


def worldmodel_figure_command(args: argparse.Namespace) -> None:
    layers = list(args.layers)
    label_a = args.label_a
    label_b = args.label_b
    probe_dir = args.probe_dir

    cell_mats: dict[int, np.ndarray] = {}
    board_mats: dict[int, np.ndarray] = {}
    raw_means: list[float] = []
    aligned_means: list[float] = []
    raw_stds: list[float] = []
    aligned_stds: list[float] = []

    focus_layer = args.focus_layer if args.focus_layer is not None else max(layers)
    tile_map_focus: np.ndarray | None = None

    for layer in layers:
        cell, board = _load_transfer_matrix(probe_dir, label_a, label_b, layer)
        cell_mats[layer] = cell
        board_mats[layer] = board

        wa = _load_probe_weight(_probe_path(probe_dir, label_a, layer))
        wb = _load_probe_weight(_probe_path(probe_dir, label_b, layer))
        raw = rowwise_cosine(wa, wb)
        wb_aligned = procrustes_align(wa, wb)
        aligned = rowwise_cosine(wa, wb_aligned)
        raw_means.append(float(raw.mean()))
        raw_stds.append(float(raw.std(ddof=1)))
        aligned_means.append(float(aligned.mean()))
        aligned_stds.append(float(aligned.std(ddof=1)))

        if layer == focus_layer:
            tile_map_focus = tilewise_mean(aligned)

    if tile_map_focus is None:
        raise ValueError(f"Focus layer {focus_layer} not available in layers {layers}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 6.5))
    gs = fig.add_gridspec(2, max(2, len(layers)), width_ratios=[1] * max(2, len(layers)), height_ratios=[1, 1.05])

    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx])
        mat = cell_mats[layer]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0)
        ax.set_title(f"Transfer cell acc. L{layer}")
        ax.set_xticks([0, 1], labels=[label_a, label_b])
        ax.set_yticks([0, 1], labels=[f"probe:{label_a}", f"probe:{label_b}"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)
        if idx == len(layers) - 1:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Cell accuracy")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(layers, raw_means, marker="o", label="raw cosine")
    ax2.plot(layers, aligned_means, marker="o", label="aligned cosine")
    ax2.fill_between(layers, np.array(raw_means) - np.array(raw_stds), np.array(raw_means) + np.array(raw_stds), alpha=0.15)
    ax2.fill_between(
        layers,
        np.array(aligned_means) - np.array(aligned_stds),
        np.array(aligned_means) + np.array(aligned_stds),
        alpha=0.15,
    )
    ax2.set_title("Probe-weight similarity")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean rowwise cosine")
    ax2.set_ylim(bottom=min(-0.05, float(min(raw_means + aligned_means)) - 0.05), top=1.02)
    ax2.grid(True, alpha=0.2)
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(gs[1, 1:])
    im2 = ax3.imshow(tile_map_focus, vmin=-1.0, vmax=1.0)
    ax3.set_title(f"Per-tile aligned cosine, L{focus_layer}")
    ax3.set_xticks(range(BOARD_DIM))
    ax3.set_yticks(range(BOARD_DIM))
    for i in range(BOARD_DIM):
        for j in range(BOARD_DIM):
            ax3.text(j, i, f"{tile_map_focus[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cbar2 = fig.colorbar(im2, ax=ax3, shrink=0.82)
    cbar2.set_label("Mean aligned cosine over tile states")

    fig.suptitle("World-model change under curious continuation", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    summary = {
        "layers": layers,
        "label_a": label_a,
        "label_b": label_b,
        "cell_transfer": {str(layer): cell_mats[layer].tolist() for layer in layers},
        "board_transfer": {str(layer): board_mats[layer].tolist() for layer in layers},
        "probe_weight_similarity": {
            "raw_mean": raw_means,
            "raw_std": raw_stds,
            "aligned_mean": aligned_means,
            "aligned_std": aligned_stds,
        },
        "focus_layer": focus_layer,
        "focus_tile_aligned_cosine": tile_map_focus.tolist(),
    }
    save_json(args.summary_out, summary)


def localization_figure_command(args: argparse.Namespace) -> None:
    compare = load_json(args.compare_summary)
    drift = load_json(args.weight_drift_summary)

    act = compare["activation_cosine"]
    layers = sorted(int(k) for k in act.keys())
    act_means = [float(act[str(layer)]["mean"]) for layer in layers]
    js_by_move = np.asarray(compare["policy"]["js_by_move"], dtype=np.float64)
    disagree_by_move = np.asarray(compare["policy"]["top1_disagreement_by_move"], dtype=np.float64)

    ratios = np.asarray(drift["ratio_matrix"], dtype=np.float64)
    rows = list(drift["rows"])
    cols = list(drift["columns"])

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, max(4.0, 0.33 * len(rows) + 1.0)), width_ratios=[1.0, 1.2, 1.4])

    axes[0].plot(layers, act_means, marker="o")
    axes[0].set_title("Activation cosine")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean cosine")
    axes[0].set_ylim(top=1.02)
    axes[0].grid(True, alpha=0.2)

    moves = np.arange(1, len(js_by_move) + 1)
    axes[1].plot(moves, js_by_move, label="JS divergence")
    axes[1].plot(moves, disagree_by_move, label="Top-1 disagreement")
    axes[1].set_title("Output divergence on shared prefixes")
    axes[1].set_xlabel("Move number")
    axes[1].set_ylabel("Divergence")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)

    im = axes[2].imshow(np.log10(ratios + 1e-12), aspect="auto")
    axes[2].set_title("Base-relative weight drift")
    axes[2].set_xticks(np.arange(len(cols)), labels=cols)
    axes[2].set_yticks(np.arange(len(rows)), labels=rows)
    plt.setp(axes[2].get_xticklabels(), rotation=25, ha="right")
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.82)
    cbar.set_label("log10 ||ΔW|| / ||W0||")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def schematic_command(args: argparse.Namespace) -> None:
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str) -> None:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=11)

    box(0.08, 0.34, 0.22, 0.3, "Shared base\ncheckpoint")
    box(0.41, 0.58, 0.22, 0.24, "Standard\ncontinuation")
    box(0.41, 0.10, 0.22, 0.24, "Curious\ncontinuation")
    box(0.74, 0.34, 0.20, 0.30, "Shared-prefix\nanalysis")

    arrowprops = dict(arrowstyle="->", linewidth=1.8)
    ax.annotate("", xy=(0.41, 0.70), xytext=(0.30, 0.52), arrowprops=arrowprops)
    ax.annotate("", xy=(0.41, 0.22), xytext=(0.30, 0.46), arrowprops=arrowprops)
    ax.annotate("", xy=(0.74, 0.49), xytext=(0.63, 0.70), arrowprops=arrowprops)
    ax.annotate("", xy=(0.74, 0.49), xytext=(0.63, 0.22), arrowprops=arrowprops)

    ax.text(0.52, 0.87, "same init, standard data", ha="center", va="center", fontsize=10)
    ax.text(0.52, 0.04, "same init, novelty/curious data", ha="center", va="center", fontsize=10)
    ax.text(0.84, 0.18, "probe transfer\nactivation similarity\nweight drift", ha="center", va="center", fontsize=10)
    ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-plot helpers for shared-init Othello continuation experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_hist = sub.add_parser("training-curves", help="Plot continuation histories from epoch log JSON files.")
    p_hist.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Repeated label/path pairs, e.g. --run fixed data/classic_fixed_e50",
    )
    p_hist.add_argument(
        "--metrics",
        nargs="+",
        default=["val_loss", "eval_top1", "train_loss", "gen_novelty"],
        help="Metrics to plot. train_loss maps across fixed/curious log formats.",
    )
    p_hist.add_argument("--epoch_origin", type=int, default=None, help="Subtract this epoch from the x-axis.")
    p_hist.add_argument("--title", type=str, default="Continuation dynamics after shared init")
    p_hist.add_argument("--summary_out", type=Path, required=True)
    p_hist.add_argument("--plot_out", type=Path, required=True)
    p_hist.set_defaults(func=training_curves_command)

    p_drift = sub.add_parser("weight-drift", help="Compare two continuation checkpoints to a shared base.")
    p_drift.add_argument("--base_ckpt", type=Path, required=True)
    p_drift.add_argument("--ckpt_a", type=Path, required=True)
    p_drift.add_argument("--ckpt_b", type=Path, required=True)
    p_drift.add_argument("--label_a", type=str, default="standard")
    p_drift.add_argument("--label_b", type=str, default="curious")
    p_drift.add_argument("--summary_out", type=Path, required=True)
    p_drift.add_argument("--plot_out", type=Path, required=True)
    p_drift.set_defaults(func=weight_drift_command)

    p_diff = sub.add_parser("weight-drift-diff", help="Plot signed curious-minus-fixed drift from an existing drift summary.")
    p_diff.add_argument("--weight_drift_summary", type=Path, required=True)
    p_diff.add_argument("--plot_out", type=Path, required=True)
    p_diff.set_defaults(func=weight_drift_difference_command)

    p_world = sub.add_parser("figure-worldmodel", help="Make the main world-model comparison figure from probe outputs.")
    p_world.add_argument("--probe_dir", type=Path, required=True)
    p_world.add_argument("--label_a", type=str, default="standard")
    p_world.add_argument("--label_b", type=str, default="curious")
    p_world.add_argument("--layers", type=int, nargs="+", default=[5, 8])
    p_world.add_argument("--focus_layer", type=int, default=None)
    p_world.add_argument("--out", type=Path, required=True)
    p_world.add_argument("--summary_out", type=Path, required=True)
    p_world.set_defaults(func=worldmodel_figure_command)

    p_loc = sub.add_parser("figure-localization", help="Make the internal-drift/localization figure.")
    p_loc.add_argument("--compare_summary", type=Path, required=True)
    p_loc.add_argument("--weight_drift_summary", type=Path, required=True)
    p_loc.add_argument("--out", type=Path, required=True)
    p_loc.set_defaults(func=localization_figure_command)

    p_schem = sub.add_parser("schematic", help="Draw the training/setup schematic.")
    p_schem.add_argument("--out", type=Path, required=True)
    p_schem.add_argument("--title", type=str, default="Shared-init continuation setup")
    p_schem.set_defaults(func=schematic_command)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
