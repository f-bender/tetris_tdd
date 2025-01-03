from typing import Protocol

from tetris.game_logic.components.board import Board


class MergeStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class MergeStrategyImpl:
    def apply(self, board: Board) -> None:
        board.merge_active_block()
