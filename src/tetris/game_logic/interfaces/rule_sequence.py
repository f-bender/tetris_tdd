from collections.abc import Sequence

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule


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
        from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
        from tetris.game_logic.rules.core.post_merge.post_merge_rule import PostMergeRule
        from tetris.game_logic.rules.core.spawn.spawn import SpawnRule

        return cls((MoveRule(), RotateRule(), SpawnRule(), DropMergeRule(), PostMergeRule()))
