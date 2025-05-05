try:
    from inputs import devices
except ImportError as e:
    msg = (
        "The Gamepad controller requires the extra `gamepad` dependency to be installed using "
        "`pip install tetris[gamepad]` or `uv sync --extra gamepad`!"
    )
    raise ImportError(msg) from e

import logging
from threading import Thread
from time import sleep

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller

LOGGER = logging.getLogger(__name__)


class GamepadController(Controller):
    SYMBOL = "🎮"

    def __init__(self, gamepad_index: int = 0) -> None:
        if gamepad_index >= (num_gamepads := len(devices.gamepads)):
            msg = f"Invalid gamepad index '{gamepad_index}' - only {num_gamepads} gamepads connected!"
            raise ValueError(msg)

        self._gamepad_index = gamepad_index
        self._up = self._down = self._left = self._right = False

        Thread(target=self.continuously_poll_gamepad, daemon=True).start()

    @property
    def symbol(self) -> str:
        return "🎮"

    def continuously_poll_gamepad(self) -> None:
        while True:
            try:
                events = devices.gamepads[self._gamepad_index].read()
            except Exception:
                LOGGER.exception("Error while polling gamepad:")
                sleep(1)
                continue

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
