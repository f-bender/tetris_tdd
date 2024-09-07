from typing import Protocol

from tetris.game_logic.components.board import Board


class BoardManipulation(Protocol):
    def manipulate(self, board: Board) -> None: ...
