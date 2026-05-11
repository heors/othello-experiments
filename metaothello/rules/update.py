from ..constants import DIRECTIONS, EMPTY
from ..metaothello import MetaOthello
from .base import UpdateRule
from .validation import is_in_board


class StandardFlankingUpdateRule(UpdateRule):
    """Standard Othello update rule that flips flanked opponent pieces."""

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and flip all flanked opponent pieces to current player's color."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for direction in DIRECTIONS:
            nx, ny = curr_x + direction[0], curr_y + direction[1]
            if not is_in_board(nx, ny) or mo.board[nx, ny] != -curr_color:
                continue

            while True:
                nx, ny = nx + direction[0], ny + direction[1]
                if not is_in_board(nx, ny) or mo.board[nx, ny] == EMPTY:
                    break
                if mo.board[nx, ny] == curr_color:
                    nx, ny = curr_x + direction[0], curr_y + direction[1]
                    while mo.board[nx, ny] == -curr_color:
                        mo.board[nx, ny] = curr_color
                        nx, ny = nx + direction[0], ny + direction[1]
                    break


class DeleteFlankingUpdateRule(UpdateRule):
    """Update rule that removes flanked opponent pieces instead of flipping them."""

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and remove all flanked opponent pieces from the board."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for direction in DIRECTIONS:
            nx, ny = curr_x + direction[0], curr_y + direction[1]
            if not is_in_board(nx, ny) or mo.board[nx, ny] != -curr_color:
                continue

            while True:
                nx, ny = nx + direction[0], ny + direction[1]
                if not is_in_board(nx, ny) or mo.board[nx, ny] == EMPTY:
                    break
                if mo.board[nx, ny] == curr_color:
                    nx, ny = curr_x + direction[0], curr_y + direction[1]
                    while mo.board[nx, ny] == -curr_color:
                        mo.board[nx, ny] = EMPTY
                        nx, ny = nx + direction[0], ny + direction[1]
                    break


class NoMiddleFlipUpdateRule(UpdateRule):
    """Update rule that only flips the first and last pieces in a flanked sequence.

    Middle pieces in the sequence remain unchanged.
    """

    @staticmethod
    def update(mo: MetaOthello, x: int, y: int) -> None:
        """Place piece and flip only the endpoints of flanked sequences."""
        curr_color = mo.next_color
        curr_x, curr_y = x, y
        mo.board[curr_x, curr_y] = curr_color

        for direction in DIRECTIONS:
            nx, ny = curr_x + direction[0], curr_y + direction[1]
            if not is_in_board(nx, ny) or mo.board[nx, ny] != -curr_color:
                continue

            flanked = [(nx, ny)]
            while True:
                nx, ny = nx + direction[0], ny + direction[1]
                if not is_in_board(nx, ny) or mo.board[nx, ny] == EMPTY:
                    break
                if mo.board[nx, ny] == curr_color:
                    if len(flanked) > 1:
                        to_flip = [flanked[0], flanked[-1]]
                        for fx, fy in to_flip:
                            mo.board[fx, fy] = curr_color
                    elif len(flanked) == 1:
                        fx, fy = flanked[0]
                        mo.board[fx, fy] = curr_color
                    break
                if mo.board[nx, ny] == -curr_color:
                    flanked.append((nx, ny))
