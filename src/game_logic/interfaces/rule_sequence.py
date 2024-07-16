from typing import Sequence

from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.rule import Rule


class RuleSequence:
    def __init__(self, rule_sequence: Sequence[Rule]) -> None:
        self._rule_sequence = rule_sequence

    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        for rule in self._rule_sequence:
            rule.apply(frame_counter, action_counter, board)
