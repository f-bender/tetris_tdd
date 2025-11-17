from typing import Protocol

from tetris.game_logic.action_counter import ActionCounter


# might be a bad name...
class RuntimeRule(Protocol):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
    ) -> None: ...
