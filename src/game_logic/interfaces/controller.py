from abc import ABC, abstractmethod
from typing import NamedTuple


class Action(NamedTuple):
    move_left: bool = False
    move_right: bool = False
    rotate_left: bool = False
    rotate_right: bool = False
    quick_drop: bool = False


class Controller(ABC):
    @abstractmethod
    def get_action(self) -> Action: ...

    # the Game class uses this to display a message what button to press in order to trigger something (e.g. start a
    # new game on game over screen)
    def get_button_description(self, action: Action) -> str:
        return " + ".join(
            action_name.replace("_", " ").title()
            for action_name, action_is_performed in action._asdict().items()
            if action_is_performed
        )
