try:
    from inputs import get_gamepad
except ImportError as e:
    msg = (
        "The Gamepad controller requires the optional `gamepad` dependency to be installed using "
        "`pip install tetris[gamepad]`!"
    )
    raise ImportError(msg) from e

from threading import Lock, Thread

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller


class GamepadController(Controller):
    def __init__(self) -> None:
        Thread(target=self.continuously_poll_gamepad, daemon=True).start()
        self._current_action = Action()
        self._lock = Lock()
        self._up = self._down = self._left = self._right = False

    def continuously_poll_gamepad(self) -> None:
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_HAT0Y":
                    self._up = self._down = False
                    if event.state > 0:
                        self._down = True
                    elif event.state < 0:
                        self._up = True
                if event.code == "ABS_HAT0X":
                    self._left = self._right = False
                    if event.state > 0:
                        self._right = True
                    elif event.state < 0:
                        self._left = True

    def get_action(self, board: Board | None = None) -> Action:
        return Action(up=self._up, down=self._down, left=self._left, right=self._right)
