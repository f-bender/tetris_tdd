from typing import Protocol

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board


class Rule(Protocol):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None: ...
