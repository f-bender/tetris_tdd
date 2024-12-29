from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller


class StubController(Controller):
    def __init__(self, action: Action) -> None:
        self._action = action

    def get_action(self, board: Board | None = None) -> Action:
        return self._action
