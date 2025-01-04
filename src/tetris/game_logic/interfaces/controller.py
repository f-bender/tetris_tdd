from abc import ABC, abstractmethod
from typing import NamedTuple, Self

from tetris.game_logic.components.board import Board


class Action(NamedTuple):
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    left_shoulder: bool = False
    right_shoulder: bool = False
    confirm: bool = False
    cancel: bool = False

    def __or__(self, other: Self) -> "Action":
        return Action(*(a or b for a, b in zip(self, other, strict=True)))


class Controller(ABC):
    @abstractmethod
    def get_action(self, board: Board | None = None) -> Action: ...

    # the Game class uses this to display a message what button to press in order to trigger something (e.g. start a
    # new game on game over screen)
    def get_button_description(self, action: Action) -> str:
        return " + ".join(
            action_name.replace("_", " ").title()
            for action_name, action_is_performed in action._asdict().items()
            if action_is_performed
        )
