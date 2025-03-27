from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller


class StubController(Controller):
    def __init__(self, action: Action) -> None:
        self._action = action

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

    def get_action(self, board: Board | None = None) -> Action:
        return self._action
