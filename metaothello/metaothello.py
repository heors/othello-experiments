from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import BLACK, BOARD_DIM, MAX_STEPS, SQUARES, WHITE, move2tuple, tuple2move
from .rules.base import InitializeBoard, UpdateRule, ValidationRule

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class MetaOthello:
    """Abstract class implementation for Othello and its variants."""

    alias: str = "base"

    def __init__(
        self,
        initialization_rule: type[InitializeBoard],
        validation_rules: list[type[ValidationRule]],
        update_rules: list[type[UpdateRule]],
    ) -> None:
        """Initialize a MetaOthello game with the given rules."""
        self.board = np.zeros((BOARD_DIM, BOARD_DIM))
        self.done = False
        self.next_color = BLACK
        self.history = []
        self.board_history = []
        # Used to cache valid moves. NOT a history of all valid moves at each timestep.
        # None = not yet computed
        self.valid_moves: list[str | None] | None = None

        # rules
        self.initialization_rule = initialization_rule
        self.validation_rules = validation_rules
        self.update_rules = update_rules

        self._initialize_board()

    def _initialize_board(self) -> None:
        self.initialization_rule.init_board(self)

    def _update_board(self, move: str | None) -> None:
        """Updates the board according to the update rules after a valid move."""
        if move is None:
            pass
        else:
            x, y = move2tuple[move]
            for rule in self.update_rules:
                rule.update(self, x, y)

    def is_valid_move(self, move: str | None) -> bool:
        """Check if a move is valid based on the current board state."""
        if move is None:
            valid = self.get_all_valid_moves()
            return len(valid) == 1 and valid[0] is None
        if self.valid_moves is not None:
            return move in self.valid_moves
        x, y = move2tuple[move]
        valid = True
        for rule in self.validation_rules:
            if not rule.is_valid(self, x, y):
                valid = False
        return valid

    def play_move(self, move: str | None, override: bool = False) -> None:
        """Plays a move for the current player."""
        # Check if the move is valid.
        if not override and not self.is_valid_move(move):
            msg = (
                f"Move {move} is invalid in current board state "
                f"(game {self.alias}, sequence {self.history}, valid moves {self.valid_moves})."
            )
            raise ValueError(msg)

        if move is not None:
            self._update_board(move)

        self.history.append(move)
        self.board_history.append(self.board.copy())
        self.next_color = -self.next_color
        self.valid_moves = None  # Reset valid next moves

    def get_all_valid_moves(self) -> list[str | None]:
        """Returns all valid moves for the current player."""
        if self.valid_moves is None:
            possible_moves = []
            for s in SQUARES:
                if self.is_valid_move(s):
                    possible_moves.append(s)

            if len(possible_moves) == 0:  # Player must pass
                possible_moves.append(None)

            self.valid_moves = possible_moves

        return self.valid_moves

    def get_random_valid_move(self) -> str | None:
        """Returns a random valid move for the current player."""
        index = list(SQUARES)
        np.random.shuffle(index)
        for s in index:
            if self.is_valid_move(s):
                return s

        return None

    def generate_random_game(self) -> None:
        """Plays a random game until termination."""
        steps = 0

        while not self.done and steps < MAX_STEPS:
            move = self.get_random_valid_move()
            self.play_move(move)

            if len(self.history) >= 2 and self.history[-1] is None and self.history[-2] is None:
                self.done = True

            steps += 1

    def recover_from_history(self, history: list[str | None]) -> None:
        """Recover the board state from a given history of moves."""
        for move in history:
            self.play_move(move)

    def print_board(self) -> None:
        """Prints the board state."""
        print("  " + " ".join([chr(ord("A") + i) for i in range(BOARD_DIM)]))
        for i in range(BOARD_DIM):
            print(
                str(i + 1)
                + " "
                + " ".join(
                    [
                        "B"
                        if self.board[i, j] == BLACK
                        else "W"
                        if self.board[i, j] == WHITE
                        else "."
                        for j in range(BOARD_DIM)
                    ]
                )
            )
        print(f"{['W', 'B'][self.next_color == BLACK]} to move.")

    def get_history(self) -> list[str | None]:
        """Returns the history of moves played so far."""
        return self.history.copy()

    def get_board_history(self) -> list[np.ndarray]:
        """Returns the history of board states so far."""
        return self.board_history.copy()

    def plot_board(
        self,
        ax: Axes | None = None,
        shading: np.ndarray | str | None = None,
        move: tuple[int, int] | None = None,
        vmin: float | None = 0,
        vmax: float | None = None,
        cmap: str = "Reds",
        annotate_cells: bool = False,
    ) -> Axes:
        """Plot the board.

        The board is shown as a grid with black or white circles in the appropriate places.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if ax is None:
            _fig, ax = plt.subplots()

        ax.set_aspect("equal")
        ax.set_xlim(0, BOARD_DIM)
        ax.set_ylim(0, BOARD_DIM)
        ax.set_xticks(np.arange(0, BOARD_DIM, 1))
        ax.set_yticks(np.arange(0, BOARD_DIM, 1))
        ax.grid(which="both")

        if isinstance(shading, str) and shading == "valid":
            shading = np.zeros(self.board.shape)
            for i in range(BOARD_DIM):
                for j in range(BOARD_DIM):
                    if self.is_valid_move(tuple2move[(i, j)]):
                        shading[i, j] = 1

        # If shading is provided, add it to the board
        if shading is not None:
            if shading.shape != self.board.shape:
                raise ValueError("Shading must be the same shape as the board")
            if vmax is None:
                vmax = np.max(shading)

            if vmin is None:
                vmin = np.min(shading)

            norm = Normalize(vmin=vmin, vmax=vmax)
            colormap = plt.get_cmap(cmap)

            for i in range(BOARD_DIM):
                for j in range(BOARD_DIM):
                    if shading[i, j] != 0:
                        rect = plt.Rectangle(
                            (j, i), 1, 1, fill=True, color=colormap(norm(shading[i, j])), alpha=0.7
                        )
                        ax.add_artist(rect)

            if annotate_cells:
                for i in range(BOARD_DIM):
                    for j in range(BOARD_DIM):
                        if not np.isnan(shading[i, j]):
                            ax.text(
                                j + 0.5,
                                i + 0.5,
                                f"{shading[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color="black",
                                fontsize=8,
                            )

        # If a move is provided, add it to the board
        if move is not None:
            move_rect = plt.Rectangle(
                (move[1], move[0]), 1, 1, fill=True, color="cornflowerblue", alpha=0.7
            )
            ax.add_artist(move_rect)

        for x, y in np.ndindex(BOARD_DIM, BOARD_DIM):
            if self.board[x, y] == BLACK:
                circle = plt.Circle((y + 0.5, x + 0.5), 0.2, color="black", ec="black", lw=1)
                ax.add_artist(circle)
            elif self.board[x, y] == WHITE:
                circle = plt.Circle((y + 0.5, x + 0.5), 0.2, color="white", ec="black", lw=1)
                ax.add_artist(circle)

        ax.invert_yaxis()
        ax.axis("off")
        # Add back the outline of the board
        outline = plt.Rectangle((0, 0), BOARD_DIM, BOARD_DIM, edgecolor="black", facecolor="none")
        ax.add_artist(outline)

        # Draw the grid
        for i in range(1, BOARD_DIM):
            ax.axhline(i, color="black", lw=0.5)
            ax.axvline(i, color="black", lw=0.5)

        # Add small circles in all the intersections of the grid for aesthetics
        # for i in range(1, BOARD_DIM):
        #     for j in range(1, BOARD_DIM):
        #         grid_circle = plt.Circle(
        #             (i + 0.025, j + 0.025), 0.05, color='black', ec='black', lw=0.5
        #         )
        #         ax.add_artist(grid_circle)

        # Add the column and row labels: A, B, C, D, E, F, G, H for the columns
        # and 1, 2, 3, 4, 5, 6, 7, 8 for the rows.
        for i in range(BOARD_DIM):
            ax.text(i + 0.5, -0.5, chr(ord("A") + i), ha="center", va="center", fontsize=12)
            ax.text(-0.5, i + 0.5, str(i + 1), ha="center", va="center", fontsize=12)

        for spine in ax.spines.values():
            spine.set_visible(True)

        return ax
