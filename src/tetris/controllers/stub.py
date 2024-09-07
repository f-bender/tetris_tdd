from tetris.game_logic.interfaces.controller import Action, Controller


class StubController(Controller):
    def __init__(self, action: Action) -> None:
        self._action = action

    def get_action(self) -> Action:
        return self._action
