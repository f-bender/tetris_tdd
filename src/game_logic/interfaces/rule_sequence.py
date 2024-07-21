from typing import Sequence

from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.rule import Rule


# Note that `RuleSequence` itself implements the `Rule` Protocol, so it could be seen as an implementation of the
# Composite pattern
class RuleSequence:
    def __init__(self, rule_sequence: Sequence[Rule]) -> None:
        self._rule_sequence = rule_sequence

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        for rule in self._rule_sequence:
            rule.apply(frame_counter, action_counter, board, callback_collection)
