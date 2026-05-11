
from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from metaothello.constants import BLACK, MAX_STEPS, WHITE, move2tuple
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.dataset import SequenceDataset
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_fresh_model, load_model_from_ckpt

logger = logging.getLogger(__name__)

VOCAB_SIZE = 66
BLOCK_SIZE = MAX_STEPS - 1
DESCRIPTOR_DIM = 66
MAX_RETRIES = 1000
BIT_MASKS = (np.uint64(1) << np.arange(64, dtype=np.uint64))


@dataclass(slots=True)
class AutocastSpec:
    enabled: bool
    device_type: str | None = None
    dtype: torch.dtype | None = None

    def context(self):
        if not self.enabled or self.device_type is None or self.dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def configure_runtime(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_autocast(device: torch.device, amp_dtype: str = "bf16", enabled: bool = True) -> AutocastSpec:
    if not enabled or device.type != "cuda":
        return AutocastSpec(enabled=False)
    if amp_dtype == "fp16":
        return AutocastSpec(enabled=True, device_type="cuda", dtype=torch.float16)
    return AutocastSpec(enabled=True, device_type="cuda", dtype=torch.bfloat16)


def parse_epoch_from_path(path: Path | None) -> int:
    if path is None:
        return 0
    m = re.search(r"epoch_(\d+)\.ckpt$", path.name)
    return int(m.group(1)) if m else 0


def raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def model_to_device(model: torch.nn.Module, device: torch.device, data_parallel: bool = True) -> torch.nn.Module:
    if device.type == "cuda" and torch.cuda.device_count() > 1 and data_parallel:
        return torch.nn.DataParallel(model).to(device)
    return model.to(device)


def save_model_ckpt(model: torch.nn.Module, ckpt_dir: Path, epoch: int) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch}.ckpt"
    torch.save(raw_model(model).state_dict(), out)
    return out


def ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        hint = ""
        if path.is_absolute():
            hint = " If you intended a workspace-local output, remove the leading '/'."
        raise PermissionError(
            f"Cannot create parent directory '{path.parent}' for output path '{path}'.{hint}"
        ) from exc


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    ez = np.exp(z)
    return ez / ez.sum()


def masked_log_probs(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid_mask, float("-inf"))
    return torch.log_softmax(masked, dim=-1)


def board_to_bitboards(board: np.ndarray) -> tuple[np.uint64, np.uint64]:
    flat = np.asarray(board).reshape(-1)
    black = BIT_MASKS[flat == BLACK].sum(dtype=np.uint64)
    white = BIT_MASKS[flat == WHITE].sum(dtype=np.uint64)
    return black, white


@dataclass(slots=True)
class StateSignature:
    black: np.uint64
    white: np.uint64
    turn: int
    bucket: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (int(self.black), int(self.white), int(self.turn), int(self.bucket))


@dataclass(slots=True)
class CandidateMove:
    move_physical: str | None
    move_token_name: str | None
    token_id: int
    board: np.ndarray
    signature: StateSignature


class KNNNoveltyArchive:
    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = int(max_size)
        self.black = np.zeros(self.max_size, dtype=np.uint64)
        self.white = np.zeros(self.max_size, dtype=np.uint64)
        self.turn = np.zeros(self.max_size, dtype=np.uint8)
        self.bucket = np.zeros(self.max_size, dtype=np.uint8)
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add_signature(self, sig: StateSignature) -> None:
        self.black[self.ptr] = sig.black
        self.white[self.ptr] = sig.white
        self.turn[self.ptr] = sig.turn
        self.bucket[self.ptr] = sig.bucket
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add_signatures(self, signatures: Iterable[StateSignature]) -> None:
        for sig in signatures:
            self.add_signature(sig)

    def score_candidates(
        self,
        candidates: Sequence[CandidateMove],
        *,
        k: int = 32,
        extra_signatures: Sequence[StateSignature] | None = None,
        **_: Any,
    ) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)

        size = self.size
        if size == 0 and not extra_signatures:
            return np.ones(len(candidates), dtype=np.float32)

        black = self.black[:size]
        white = self.white[:size]
        turn = self.turn[:size].astype(np.int16, copy=False)
        bucket = self.bucket[:size].astype(np.int16, copy=False)

        if extra_signatures:
            e_black = np.array([sig.black for sig in extra_signatures], dtype=np.uint64)
            e_white = np.array([sig.white for sig in extra_signatures], dtype=np.uint64)
            e_turn = np.array([sig.turn for sig in extra_signatures], dtype=np.int16)
            e_bucket = np.array([sig.bucket for sig in extra_signatures], dtype=np.int16)
            black = np.concatenate([black, e_black], axis=0)
            white = np.concatenate([white, e_white], axis=0)
            turn = np.concatenate([turn, e_turn], axis=0)
            bucket = np.concatenate([bucket, e_bucket], axis=0)

        c_black = np.array([c.signature.black for c in candidates], dtype=np.uint64)
        c_white = np.array([c.signature.white for c in candidates], dtype=np.uint64)
        c_turn = np.array([c.signature.turn for c in candidates], dtype=np.int16)
        c_bucket = np.array([c.signature.bucket for c in candidates], dtype=np.int16)

        occ_archive = (black | white)[None, :]
        occ_candidates = (c_black | c_white)[:, None]
        occ_diff = np.bitwise_count(occ_archive ^ occ_candidates).astype(np.int16, copy=False)
        color_swap = np.bitwise_count((black[None, :] & c_white[:, None]) | (white[None, :] & c_black[:, None])).astype(np.int16, copy=False)
        turn_diff = np.abs(turn[None, :] - c_turn[:, None])
        bucket_diff = np.abs(bucket[None, :] - c_bucket[:, None])
        d = occ_diff + 2 * color_swap + turn_diff + bucket_diff

        kk = min(int(k), d.shape[1])
        nearest = np.partition(d, kk - 1, axis=1)[:, :kk]
        return nearest.mean(axis=1, dtype=np.float32) / DESCRIPTOR_DIM

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": np.array("knn"),
            "max_size": np.array(self.max_size),
            "size": np.array(self.size),
            "ptr": np.array(self.ptr),
            "black": self.black[: self.size].copy(),
            "white": self.white[: self.size].copy(),
            "turn": self.turn[: self.size].copy(),
            "bucket": self.bucket[: self.size].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.max_size = int(state["max_size"])
        self.black = np.zeros(self.max_size, dtype=np.uint64)
        self.white = np.zeros(self.max_size, dtype=np.uint64)
        self.turn = np.zeros(self.max_size, dtype=np.uint8)
        self.bucket = np.zeros(self.max_size, dtype=np.uint8)
        self.size = int(state["size"])
        self.ptr = int(state["ptr"])
        self.black[: self.size] = state["black"]
        self.white[: self.size] = state["white"]
        self.turn[: self.size] = state["turn"]
        self.bucket[: self.size] = state["bucket"]


class CountNoveltyArchive:
    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = int(max_size)
        self.queue: deque[tuple[int, int, int, int]] = deque()
        self.counts: Counter[tuple[int, int, int, int]] = Counter()

    def __len__(self) -> int:
        return len(self.queue)

    def add_signature(self, sig: StateSignature) -> None:
        key = sig.key
        if len(self.queue) >= self.max_size:
            old = self.queue.popleft()
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]
        self.queue.append(key)
        self.counts[key] += 1

    def add_signatures(self, signatures: Iterable[StateSignature]) -> None:
        for sig in signatures:
            self.add_signature(sig)

    def score_candidates(
        self,
        candidates: Sequence[CandidateMove],
        *,
        extra_signatures: Sequence[StateSignature] | None = None,
        **_: Any,
    ) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        local_counts: Counter[tuple[int, int, int, int]] = Counter(sig.key for sig in (extra_signatures or []))
        bonuses = []
        for cand in candidates:
            count = self.counts.get(cand.signature.key, 0) + local_counts.get(cand.signature.key, 0)
            bonuses.append(1.0 / math.sqrt(count + 1.0))
        return np.asarray(bonuses, dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        q = np.array(list(self.queue), dtype=np.uint64) if self.queue else np.zeros((0, 4), dtype=np.uint64)
        return {
            "kind": np.array("count"),
            "max_size": np.array(self.max_size),
            "queue": q,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.max_size = int(state["max_size"])
        q = np.asarray(state["queue"], dtype=np.uint64)
        self.queue = deque(tuple(int(v) for v in row) for row in q)
        self.counts = Counter(self.queue)


NoveltyArchive = KNNNoveltyArchive | CountNoveltyArchive


def make_novelty_archive(novelty: str, archive_size: int) -> NoveltyArchive:
    novelty = novelty.lower()
    if novelty == "knn":
        return KNNNoveltyArchive(max_size=archive_size)
    if novelty == "count":
        return CountNoveltyArchive(max_size=archive_size)
    raise ValueError(f"Unknown novelty type: {novelty}")


def save_archive(archive: NoveltyArchive, path: Path) -> None:
    ensure_parent_dir(path)
    np.savez_compressed(path, **archive.state_dict())


def load_archive(path: Path, novelty: str, archive_size: int) -> NoveltyArchive:
    archive = make_novelty_archive(novelty, archive_size)
    if not path.exists():
        return archive
    with np.load(path, allow_pickle=False) as state:
        archive.load_state_dict({k: state[k] for k in state.files})
    return archive


def build_signature(board: np.ndarray, ply_after_move: int) -> StateSignature:
    black, white = board_to_bitboards(board)
    return StateSignature(black=black, white=white, turn=ply_after_move % 2, bucket=ply_after_move // 10)


def board_after_move(game: Any, move: str | None) -> np.ndarray:
    if move is None:
        return np.asarray(game.board, dtype=np.int8).copy()
    scratch = SimpleNamespace(board=np.asarray(game.board).copy(), next_color=game.next_color)
    x, y = move2tuple[move]
    for rule in game.update_rules:
        rule.update(scratch, x, y)
    return np.asarray(scratch.board, dtype=np.int8)


def build_candidates(game: Any, tok: Tokenizer) -> list[CandidateMove]:
    has_mapping = hasattr(game, "mapping")
    legal_physical = game.get_all_valid_moves()
    ply_after_move = len(game.history) + 1
    out: list[CandidateMove] = []
    for move_physical in legal_physical:
        move_token_name = game.mapping[move_physical] if has_mapping else move_physical
        board = board_after_move(game, move_physical)
        signature = build_signature(board, ply_after_move)
        out.append(
            CandidateMove(
                move_physical=move_physical,
                move_token_name=move_token_name,
                token_id=tok.stoi[move_token_name],
                board=board,
                signature=signature,
            )
        )
    return out


@torch.inference_mode()
def next_move_logits(
    model: torch.nn.Module,
    token_history: Sequence[int],
    *,
    autocast: AutocastSpec,
) -> torch.Tensor:
    device = next(raw_model(model).parameters()).device
    x = torch.tensor([list(token_history)], dtype=torch.long, device=device)
    with autocast.context():
        logits, _ = model(x)
    return logits[0, x.shape[1] - 1].float()


@torch.inference_mode()
def rollout_one_curious_game(
    model: torch.nn.Module,
    tok: Tokenizer,
    archive: NoveltyArchive,
    *,
    game_alias: str = "classic",
    beta: float = 0.25,
    tau: float = 1.0,
    k: int = 32,
    seed_first_move_random: bool = True,
    autocast: AutocastSpec | None = None,
    max_retries: int = MAX_RETRIES,
) -> tuple[list[int], np.ndarray, dict[str, float]]:
    if autocast is None:
        autocast = resolve_autocast(get_device())

    game_class = GAME_REGISTRY[game_alias]

    for _ in range(max_retries):
        game = game_class()  # type: ignore[operator]
        local_signatures: list[StateSignature] = []
        local_boards: list[np.ndarray] = []
        local_novelties: list[float] = []
        token_history: list[int] = []
        has_mapping = hasattr(game, "mapping")

        legal0 = game.get_all_valid_moves()
        first_physical = random.choice(legal0) if seed_first_move_random else legal0[0]
        first_token_name = game.mapping[first_physical] if has_mapping else first_physical
        game.play_move(first_physical)
        token_history.append(tok.stoi[first_token_name])
        first_board = np.asarray(game.board, dtype=np.int8).copy()
        first_signature = build_signature(first_board, len(game.history))
        local_signatures.append(first_signature)
        local_boards.append(first_board)

        while len(game.history) < MAX_STEPS:
            if len(game.history) >= 2 and game.history[-1] is None and game.history[-2] is None:
                break

            candidates = build_candidates(game, tok)
            logits = next_move_logits(model, token_history, autocast=autocast)
            token_ids = [cand.token_id for cand in candidates]
            model_scores = logits[token_ids].detach().cpu().numpy().astype(np.float32) / max(tau, 1e-6)
            novelty_bonus = archive.score_candidates(candidates, k=k)
            scores = model_scores + beta * novelty_bonus
            probs = softmax_numpy(scores)
            idx = int(np.random.choice(len(candidates), p=probs))
            chosen = candidates[idx]

            game.play_move(chosen.move_physical)
            token_history.append(chosen.token_id)
            local_signatures.append(chosen.signature)
            local_boards.append(chosen.board)
            local_novelties.append(float(novelty_bonus[idx]))

        if len(game.history) == MAX_STEPS:
            archive.add_signatures(local_signatures)
            seq = tok.pad_sequence(token_history, max_length=MAX_STEPS)
            boards = np.asarray(local_boards, dtype=np.int8)
            stats = {
                "novelty_mean": float(np.mean(local_novelties)) if local_novelties else 0.0,
                "novelty_last": float(local_novelties[-1]) if local_novelties else 0.0,
                "num_moves": float(len(token_history)),
            }
            return seq, boards, stats

    raise RuntimeError(f"Could not generate a valid {MAX_STEPS}-step curious game after {max_retries} attempts")


@torch.inference_mode()
def generate_curious_corpus(
    model: torch.nn.Module,
    tok: Tokenizer,
    archive: NoveltyArchive,
    *,
    num_games: int,
    game_alias: str = "classic",
    beta: float = 0.25,
    tau: float = 1.0,
    k: int = 32,
    seed_first_move_random: bool = True,
    autocast: AutocastSpec | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if autocast is None:
        autocast = resolve_autocast(get_device())

    was_training = model.training
    model.eval()

    iterator: Iterable[int]
    if progress:
        try:
            from tqdm import trange
            iterator = trange(num_games, desc=f"Generating curious {game_alias}")
        except Exception:
            iterator = range(num_games)
    else:
        iterator = range(num_games)

    seqs: list[list[int]] = []
    boards: list[np.ndarray] = []
    novelty_means: list[float] = []
    novelty_last: list[float] = []

    for _ in iterator:
        seq, board_hist, stats = rollout_one_curious_game(
            model,
            tok,
            archive,
            game_alias=game_alias,
            beta=beta,
            tau=tau,
            k=k,
            seed_first_move_random=seed_first_move_random,
            autocast=autocast,
        )
        seqs.append(seq)
        boards.append(board_hist)
        novelty_means.append(stats["novelty_mean"])
        novelty_last.append(stats["novelty_last"])

    if was_training:
        model.train()

    return (
        np.asarray(seqs, dtype=np.int32),
        np.asarray(boards, dtype=np.int8),
        {
            "num_games": float(num_games),
            "novelty_mean": float(np.mean(novelty_means)) if novelty_means else 0.0,
            "novelty_last": float(np.mean(novelty_last)) if novelty_last else 0.0,
            "archive_size": float(len(archive)),
        },
    )


def write_corpus_zarr(seqs: np.ndarray, boards: np.ndarray, out_path: Path) -> None:
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError("xarray with zarr support is required to write Zarr corpora") from exc

    ds = xr.Dataset(
        data_vars={
            "seqs": (["game", "move"], seqs.astype(np.int32)),
            "board_state": (["game", "move", "x", "y"], boards.astype(np.int8)),
        }
    )
    ensure_parent_dir(out_path)
    ds.to_zarr(out_path, mode="w", zarr_format=2)


class ZarrSequenceSampler:
    def __init__(
        self,
        zarr_path: Path,
        *,
        val_games: int = 10_000,
        seed: int = 42,
        val_idx: np.ndarray | None = None,
    ) -> None:
        try:
            import xarray as xr
        except Exception as exc:
            raise RuntimeError("xarray with zarr support is required to sample Zarr corpora") from exc

        self.zarr_path = Path(zarr_path)
        self.ds = xr.open_zarr(self.zarr_path)
        self.n = int(self.ds.sizes["game"])
        if val_idx is None:
            rng = np.random.default_rng(seed)
            if not 0 < val_games < self.n:
                raise ValueError(f"val_games must be in [1, {self.n - 1}], got {val_games}")
            self.val_idx = np.sort(rng.choice(self.n, size=val_games, replace=False)).astype(np.int64)
        else:
            self.val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
        mask = np.ones(self.n, dtype=bool)
        mask[self.val_idx] = False
        self.train_pool = np.arange(self.n, dtype=np.int64)[mask]

    def sample_train(self, num_games: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if num_games > len(self.train_pool):
            raise ValueError(f"Requested {num_games} train games but only {len(self.train_pool)} are available")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.train_pool, size=num_games, replace=False)).astype(np.int64)
        seqs = self.ds["seqs"].isel(game=idx).values.astype(np.int32)
        return seqs, idx

    def load_val(self) -> np.ndarray:
        return self.ds["seqs"].isel(game=self.val_idx).values.astype(np.int32)


def make_sequence_dataset(seqs: np.ndarray) -> SequenceDataset:
    return SequenceDataset(seqs, Tokenizer(), tokenize=False)


def compute_lr(tokens_seen: int, *, base_lr: float, warmup_tokens: int, final_tokens: int, lr_decay: bool) -> float:
    if not lr_decay:
        return base_lr
    if tokens_seen < warmup_tokens:
        mult = float(tokens_seen) / float(max(1, warmup_tokens))
    else:
        progress = float(tokens_seen - warmup_tokens) / float(max(1, final_tokens - warmup_tokens))
        progress = min(max(progress, 0.0), 1.0)
        mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return base_lr * mult


def train_one_pass(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seqs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    tokens_seen: int,
    base_lr: float,
    warmup_tokens: int,
    final_tokens: int,
    lr_decay: bool,
    grad_clip: float,
    num_workers: int = 0,
    autocast: AutocastSpec | None = None,
    log_interval: int = 50,
    log_callback: Any | None = None,
) -> dict[str, Any]:
    if autocast is None:
        autocast = resolve_autocast(device)
    dataset = make_sequence_dataset(seqs)
    loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=device.type == "cuda",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=autocast.enabled and autocast.dtype == torch.float16)
    losses: list[float] = []
    last_lr = base_lr

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        with autocast.context():
            _logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model returned no loss during training")
            loss = loss.mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tokens_seen += int((y != 0).sum().item())
        last_lr = compute_lr(
            tokens_seen,
            base_lr=base_lr,
            warmup_tokens=warmup_tokens,
            final_tokens=final_tokens,
            lr_decay=lr_decay,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = last_lr

        loss_val = float(loss.item())
        losses.append(loss_val)
        if log_callback is not None and (step == 1 or step % log_interval == 0 or step == len(loader)):
            log_callback(
                {
                    "step_in_pass": step,
                    "train_loss_step": loss_val,
                    "lr": last_lr,
                    "tokens_seen": tokens_seen,
                }
            )

    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "tokens_seen": tokens_seen,
        "last_lr": last_lr,
        "num_steps": len(loader),
        "num_examples": len(dataset),
    }


@torch.inference_mode()
def evaluate_sequence_loss(
    model: torch.nn.Module,
    seqs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    autocast: AutocastSpec | None = None,
) -> float:
    if autocast is None:
        autocast = resolve_autocast(device)
    dataset = make_sequence_dataset(seqs)
    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=device.type == "cuda",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_examples = 0
    for x, y in loader:
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        with autocast.context():
            _logits, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model returned no loss during evaluation")
            loss = loss.mean()
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_examples += bs

    if was_training:
        model.train()
    return total_loss / max(1, total_examples)


def valid_masks_from_token_seqs(seqs: np.ndarray, *, game_alias: str, tok: Tokenizer) -> np.ndarray:
    game_class = GAME_REGISTRY[game_alias]
    valid_masks = np.zeros((seqs.shape[0], MAX_STEPS, tok.vocab_size), dtype=bool)

    for i in range(seqs.shape[0]):
        game = game_class()  # type: ignore[operator]
        has_mapping = hasattr(game, "mapping")
        token_moves = tok.decode(seqs[i].tolist())

        for step in range(MAX_STEPS):
            legal_physical = game.get_all_valid_moves()
            legal_token_names = [game.mapping[m] if has_mapping else m for m in legal_physical]
            for move_name in legal_token_names:
                valid_masks[i, step, tok.stoi[move_name]] = True

            move_token = token_moves[step]
            if move_token == tok.PAD_TOKEN:
                break
            move_physical = game.reverse_mapping[move_token] if has_mapping else move_token
            game.play_move(move_physical)

    return valid_masks


@torch.inference_mode()
def evaluate_teacher_forced(
    model: torch.nn.Module,
    seqs: np.ndarray,
    valid_masks: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    autocast: AutocastSpec | None = None,
) -> dict[str, Any]:
    if autocast is None:
        autocast = resolve_autocast(device)
    was_training = model.training
    model.eval()

    num_targets = 0
    correct = 0
    prob_sum = 0.0
    nll_sum = 0.0
    t = seqs.shape[1] - 1
    per_count = np.zeros(t, dtype=np.int64)
    per_correct = np.zeros(t, dtype=np.float64)
    per_prob = np.zeros(t, dtype=np.float64)

    for start in range(0, seqs.shape[0], batch_size):
        end = min(start + batch_size, seqs.shape[0])
        x = torch.tensor(seqs[start:end, :-1], dtype=torch.long, device=device)
        y = torch.tensor(seqs[start:end, 1:], dtype=torch.long, device=device)
        valid = torch.tensor(valid_masks[start:end, 1:, :], dtype=torch.bool, device=device)

        with autocast.context():
            logits, _ = model(x)
        logp = masked_log_probs(logits.float(), valid)
        preds = logp.argmax(dim=-1)
        tgt_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        nonpad = y != 0
        correct_mask = (preds == y) & nonpad
        tgt_prob = tgt_logp.exp() * nonpad

        num_targets += int(nonpad.sum().item())
        correct += int(correct_mask.sum().item())
        prob_sum += float(tgt_prob.sum().item())
        nll_sum += float(((-tgt_logp) * nonpad).sum().item())

        per_count += nonpad.sum(dim=0).cpu().numpy().astype(np.int64)
        per_correct += correct_mask.sum(dim=0).cpu().numpy().astype(np.float64)
        per_prob += tgt_prob.sum(dim=0).cpu().numpy().astype(np.float64)

    if was_training:
        model.train()

    top1 = correct / max(1, num_targets)
    correct_prob = prob_sum / max(1, num_targets)
    nll = nll_sum / max(1, num_targets)
    per_move_top1 = np.divide(per_correct, np.maximum(per_count, 1), dtype=np.float64)
    per_move_prob = np.divide(per_prob, np.maximum(per_count, 1), dtype=np.float64)

    return {
        "top1": float(top1),
        "correct_prob": float(correct_prob),
        "nll": float(nll),
        "perplexity": float(math.exp(nll)) if nll < 20 else float("inf"),
        "num_targets": int(num_targets),
        "predicted_move_numbers": list(range(2, MAX_STEPS + 1)),
        "first_move_excluded": True,
        "per_move_top1": per_move_top1.tolist(),
        "per_move_correct_prob": per_move_prob.tolist(),
    }


@torch.inference_mode()
def evaluate_random_games(
    model: torch.nn.Module,
    *,
    game_alias: str,
    num_games: int,
    batch_size: int,
    device: torch.device,
    autocast: AutocastSpec | None = None,
) -> dict[str, Any]:
    from metaothello.analysis_utils import gen_games

    tok = Tokenizer()
    seqs, valid_masks = gen_games(game_alias, num_games=num_games, tokenizer=tok)
    return evaluate_teacher_forced(model, seqs, valid_masks, batch_size=batch_size, device=device, autocast=autocast)


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: dict[str, Any],
):
    if not enabled:
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is not installed but logging was requested") from exc
    return wandb.init(project=project, name=run_name, config=config)


def wandb_log(run: Any, payload: dict[str, Any], *, step: int | None = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def load_training_model(
    *,
    init_ckpt: Path | None,
    ckpt_dir: Path,
    device: torch.device,
    data_parallel: bool = True,
) -> tuple[torch.nn.Module, int, Path | None]:
    last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    if last_ckpt is not None:
        model = load_model_from_ckpt(last_ckpt, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
        model = model_to_device(model, device, data_parallel=data_parallel)
        return model, last_epoch, last_ckpt
    if init_ckpt is not None:
        model = load_model_from_ckpt(init_ckpt, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
        model = model_to_device(model, device, data_parallel=data_parallel)
        return model, parse_epoch_from_path(init_ckpt), init_ckpt
    model = load_fresh_model(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
    model = model_to_device(model, device, data_parallel=data_parallel)
    return model, 0, None


def load_eval_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = load_model_from_ckpt(ckpt_path, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
    return model_to_device(model, device, data_parallel=False)


def make_optimizer_config(args: Any) -> SimpleNamespace:
    return SimpleNamespace(
        learning_rate=float(args.learning_rate),
        betas=(float(args.beta1), float(args.beta2)),
        grad_norm_clip=float(args.grad_clip),
        weight_decay=float(args.weight_decay),
    )


def save_training_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_training_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")
