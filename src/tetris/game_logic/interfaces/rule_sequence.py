from collections.abc import Sequence

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.rule import Rule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule


# Note that `RuleSequence` itself implements the `Rule` Protocol, so it could be seen as an implementation of the
# Composite pattern
class RuleSequence:
    def __init__(self, rule_sequence: Sequence[Rule]) -> None:
        self._rule_sequence = rule_sequence

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        for rule in self._rule_sequence:
            rule.apply(frame_counter, action_counter, board)

    @classmethod
    def standard(cls) -> "RuleSequence":
        # Avoid circular import
        from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
        from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule

        return cls((MoveRule(), RotateRule(), SpawnDropMergeRule(), ClearFullLinesRule()))
