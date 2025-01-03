from typing import Protocol

from tetris.game_logic.components.board import Board


class DropStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class DropStrategyImpl:
    def apply(self, board: Board) -> None:
        board.drop_active_block()
