"""Abstract base classes for Othello game rules."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..metaothello import MetaOthello


class InitializeBoard(ABC):
    """Abstract base class for board initialization rules."""

    @staticmethod
    @abstractmethod
    def init_board(mo: "MetaOthello") -> None:
        """Initialize the board with starting pieces."""
        pass


class ValidationRule(ABC):
    """Abstract base class for move validation rules."""

    @staticmethod
    @abstractmethod
    def is_valid(mo: "MetaOthello", x: int, y: int) -> bool:
        """Check if a move at (x, y) is valid."""
        pass


class UpdateRule(ABC):
    """Abstract base class for board update rules."""

    @staticmethod
    @abstractmethod
    def update(mo: "MetaOthello", x: int, y: int) -> None:
        """Update the board after a move at (x, y)."""
        pass
