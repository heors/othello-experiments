from __future__ import annotations

from .constants import SQUARES
from .metaothello import MetaOthello
from .rules.initialization import ClassicInitialization, OpenSpreadInitialization
from .rules.update import (
    DeleteFlankingUpdateRule,
    NoMiddleFlipUpdateRule,
    StandardFlankingUpdateRule,
)
from .rules.validation import (
    AvailableRule,
    NeighborValidationRule,
    StandardFlankingValidationRule,
)


class ClassicOthello(MetaOthello):
    """Classic Othello game with standard rules."""

    alias = "classic"

    def __init__(self) -> None:
        """Initialize Classic Othello game."""
        super().__init__(
            initialization_rule=ClassicInitialization,
            validation_rules=[AvailableRule, StandardFlankingValidationRule],
            update_rules=[StandardFlankingUpdateRule],
        )


class NoMiddleFlip(MetaOthello):
    """Othello variant where only the endpoints of flanked sequences are flipped."""

    alias = "nomidflip"

    def __init__(self) -> None:
        """Initialize NoMiddleFlip Othello game."""
        super().__init__(
            initialization_rule=ClassicInitialization,
            validation_rules=[AvailableRule, StandardFlankingValidationRule],
            update_rules=[NoMiddleFlipUpdateRule],
        )


class DeleteFlanking(MetaOthello):
    """Othello variant where flanked pieces are deleted instead of flipped."""

    alias = "delflank"

    def __init__(self) -> None:
        """Initialize DeleteFlanking Othello game."""
        super().__init__(
            initialization_rule=OpenSpreadInitialization,
            validation_rules=[AvailableRule, NeighborValidationRule],
            update_rules=[DeleteFlankingUpdateRule],
        )


class Iago(MetaOthello):
    """Othello variant with shuffled move encoding."""

    alias = "iago"

    def __init__(self) -> None:
        """Initialize Iago Othello game with shuffled move encoding."""
        super().__init__(
            initialization_rule=ClassicInitialization,
            validation_rules=[AvailableRule, StandardFlankingValidationRule],
            update_rules=[StandardFlankingUpdateRule],
        )
        self.moves = [*SQUARES, None]
        self.shuffled_moves = self._shuffle_moves()
        self.mapping = dict(zip(self.moves, self.shuffled_moves, strict=True))
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def _shuffle_moves(self) -> list[str | None]:
        """Shuffle the piece encoding in a twisted way."""
        n = len(self.moves)
        step = 47
        offset = 13

        return [self.moves[(i * step + offset) % n] for i in range(n)]

    def get_history(self) -> list[str | None]:
        """Return shuffled history of moves."""
        return [self.mapping[move] for move in self.history]

    def recover_from_history(self, history: list[str | None]) -> None:
        """Recover the board state from a given history of moves."""
        for move in history:
            self.play_move(self.reverse_mapping[move])


GAME_REGISTRY: dict[str, type[MetaOthello]] = {
    cls.alias: cls for cls in [ClassicOthello, NoMiddleFlip, DeleteFlanking, Iago]
}
