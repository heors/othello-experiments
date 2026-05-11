"""Linear probe model and trainer for board-state prediction from GPT activations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metaothello.constants import BOARD_DIM

logger = logging.getLogger(__name__)

# Board cell categories: BLACK (-1) → 0, EMPTY (0) → 1, WHITE (1) → 2.
# The +1 offset maps board values {-1, 0, 1} to class indices {0, 1, 2}.
_CATEGORY_OFFSET = 1
_NUM_CATEGORIES = 3


class LinearProbe(nn.Module):
    """Single linear layer mapping GPT activations to board-state predictions.

    Predicts the occupancy state of each board square independently. Output
    logits have shape ``(batch_size, num_tasks, num_categories)``.
    """

    def __init__(
        self,
        device: torch.device | str,
        input_dim: int = 512,
        num_tasks: int = BOARD_DIM * BOARD_DIM,
        num_categories: int = _NUM_CATEGORIES,
    ) -> None:
        """Initialize the linear probe.

        Args:
            device: Target device for the model.
            input_dim: Dimensionality of the input activations (d_model).
            num_tasks: Number of board squares to predict (default: 64 = BOARD_DIM²).
            num_categories: Number of occupancy classes — BLACK, EMPTY, WHITE (default: 3).
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.proj = nn.Linear(self.input_dim, self.num_tasks * self.num_categories)
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optionally the cross-entropy loss.

        Args:
            x: Input activations of shape ``(batch_size, input_dim)``.
            y: Target board states of shape ``(batch_size, num_tasks)`` with
                values in ``{-1, 0, 1}`` (BLACK, EMPTY, WHITE). When None,
                loss is not computed.

        Returns:
            Tuple of ``(logits, loss)``. Logits have shape
            ``(batch_size, num_tasks, num_categories)``. Loss is None when
            y is None.
        """
        logits = self.proj(x).reshape(-1, self.num_tasks, self.num_categories)
        if y is None:
            return logits, None
        # Shift board values {-1, 0, 1} to class indices {0, 1, 2}.
        targets = (y + _CATEGORY_OFFSET).to(torch.long)
        loss = F.cross_entropy(logits.view(-1, self.num_categories), targets.view(-1))
        return logits, loss


class ProbeTrainer:
    """Training loop for LinearProbe models.

    Handles device placement (CUDA / MPS / CPU), gradient management, and
    per-epoch loss tracking. Does not save checkpoints — call
    ``torch.save(model.state_dict(), path)`` externally when needed.
    """

    def __init__(
        self,
        model: LinearProbe,
        train_dataset: Any,
        test_dataset: Any | None,
        config: dict,
    ) -> None:
        """Initialize ProbeTrainer and place the model on the appropriate device.

        Args:
            model: LinearProbe instance to train.
            train_dataset: Training dataset of ``(activations, board_states)`` tensors.
            test_dataset: Evaluation dataset, or None to skip evaluation.
            config: Training hyperparameter dict with keys:
                ``lr``, ``wd``, ``betas``, ``batch_size``, ``num_workers``,
                ``grad_norm_clip``, ``max_epochs``.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        logger.info("probe trainer config: %s", config)

        self.loss_train: list[dict] = []
        self.loss_test: list[dict] = []

        if torch.cuda.is_available():
            self.device: torch.device | int = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        elif torch.mps.is_available():
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def train(self) -> None:
        """Run the full training loop for ``config['max_epochs']`` epochs."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["wd"],
            betas=tuple(self.config.get("betas", (0.9, 0.95))),
        )

        best_loss = float("inf")

        def run_epoch(split: str) -> float:
            """Run one epoch of training or evaluation.

            Args:
                split: Either ``'train'`` or ``'test'``.

            Returns:
                Mean loss over all batches in the epoch.
            """
            is_train = split == "train"
            self.model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                batch_size=self.config["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=self.config["num_workers"],
            )
            losses: list[float] = []
            pbar = tqdm(enumerate(loader), total=len(loader))

            for i, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    _logits, loss = self.model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    self.model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["grad_norm_clip"]
                    )
                    optimizer.step()
                    self.loss_train.append(
                        {"epoch": epoch + 1, "iter": i, "loss": float(np.mean(losses))}
                    )
                else:
                    self.loss_test.append(
                        {"epoch": epoch + 1, "iter": i, "loss": float(np.mean(losses))}
                    )

                pbar.set_description(
                    f"epoch {epoch + 1} iter {i}: {split} loss {np.mean(losses):.4f}"
                )

            return float(np.mean(losses))

        for epoch in range(self.config["max_epochs"]):
            run_epoch("train")
            if self.test_dataset is not None:
                test_loss = run_epoch("test")
                logger.info("epoch %d test loss: %.4f", epoch + 1, test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss

        logger.info("Training complete. Best test loss: %.4f", best_loss)
