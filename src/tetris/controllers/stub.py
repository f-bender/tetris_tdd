from typing import Literal

from tetris.game_logic.interfaces.controller import Action, Controller


class StubController(Controller):
    def __init__(
        self, action: Action, mode: Literal["hold", "press_repeatedly"] = "hold", press_duration_frames: int = 1
    ) -> None:
        self._action = action
        self._hold = mode == "hold"
        self._press_counter = 0
        self._press_duration_frames = press_duration_frames

    @property
    def symbol(self) -> str:
        return self.get_button_description(self._action)

    def get_button_description(self, action: Action) -> str:
        description = "+".join(
            action_name[0].upper()
            for action_name, action_is_performed in action._asdict().items()
            if action_is_performed
        )
        return description or "<nothing>"

    def get_action(self) -> Action:
        if self._hold:
            return self._action

        self._press_counter += 1
        if self._press_counter >= self._press_duration_frames:
            self._press_counter = 0
            return Action()

        return self._action
