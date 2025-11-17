from collections.abc import Sequence

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.interfaces.runtime_rule import RuntimeRule


# Note that `RuntimeRuleSequence` itself implements the `RuntimeRule` Protocol, so it could be seen as an implementation
# of the Composite pattern
class RuntimeRuleSequence:
    def __init__(self, rule_sequence: Sequence[RuntimeRule]) -> None:
        self._rule_sequence = rule_sequence

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
    ) -> None:
        for rule in self._rule_sequence:
            rule.apply(frame_counter, action_counter)
