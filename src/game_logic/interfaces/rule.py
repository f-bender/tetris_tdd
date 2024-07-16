from typing import Protocol
from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board


class Rule(Protocol):
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None: ...
