import numpy as np

from ..constants import BLACK, BOARD_DIM, WHITE
from ..metaothello import MetaOthello
from .base import InitializeBoard


class ClassicInitialization(InitializeBoard):
    """Classic Othello initialization with 4 pieces in the center.

    Starting position: W[d4, e5], B[e4, d5].
    Used for Classic and Nomidflip.
    """

    @staticmethod
    def init_board(mo: MetaOthello) -> None:
        """Initialize board with classic Othello starting position."""
        mo.board = np.zeros((BOARD_DIM, BOARD_DIM))
        mo.board[3, 3] = WHITE
        mo.board[3, 4] = BLACK
        mo.board[4, 3] = BLACK
        mo.board[4, 4] = WHITE


class OpenSpreadInitialization(InitializeBoard):
    """Open spread initialization with 4 pieces spread across the board.

    Starting position: W[c6, f3], B[c3, f6].
    Used for Delflank.
    """

    @staticmethod
    def init_board(mo: MetaOthello) -> None:
        """Initialize board with open spread starting position."""
        mo.board = np.zeros((BOARD_DIM, BOARD_DIM))
        mo.board[2, 5] = WHITE
        mo.board[2, 2] = BLACK
        mo.board[5, 2] = WHITE
        mo.board[5, 5] = BLACK
