try:
    from inputs import devices
except ImportError as e:
    msg = (
        "The Gamepad controller requires the extra `gamepad` dependency to be installed using "
        "`pip install tetris[gamepad]` or `uv sync --extra gamepad`!"
    )
    raise ImportError(msg) from e

import logging
from dataclasses import asdict, dataclass
from threading import Thread
from time import sleep

from tetris.game_logic.interfaces.controller import Action, Controller

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Action:
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    left_shoulder: bool = False
    right_shoulder: bool = False
    confirm: bool = False
    cancel: bool = False


class GamepadController(Controller):
    SYMBOL = "🎮"

    def __init__(self, gamepad_index: int = 0) -> None:
        if gamepad_index >= (num_gamepads := len(devices.gamepads)):
            msg = f"Invalid gamepad index '{gamepad_index}' - only {num_gamepads} gamepads connected!"
            raise ValueError(msg)

        self._gamepad_index = gamepad_index
        self._current_action = _Action()

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
                self._handle_gamepad_event(event.code, event.state)

    def _handle_gamepad_event(self, code: str, state: int) -> None:  # noqa: C901, PLR0912
        match (code, state):
            # up/down on the D-Pad
            case ("ABS_HAT0Y", 1):
                self._current_action.up = False
                self._current_action.down = True
            case ("ABS_HAT0Y", -1):
                self._current_action.up = True
                self._current_action.down = False
            case ("ABS_HAT0Y", 0):
                self._current_action.up = False
                self._current_action.down = False
            # left/right on the D-Pad
            case ("ABS_HAT0X", 1):
                self._current_action.left = False
                self._current_action.right = True
            case ("ABS_HAT0X", -1):
                self._current_action.left = True
                self._current_action.right = False
            case ("ABS_HAT0X", 0):
                self._current_action.left = False
                self._current_action.right = False
            # A
            case ("BTN_SOUTH", 1):
                self._current_action.confirm = True
            case ("BTN_SOUTH", 0):
                self._current_action.confirm = False
            # B
            case ("BTN_EAST", 1):
                self._current_action.cancel = True
            case ("BTN_EAST", 0):
                self._current_action.cancel = False
            # LB
            case ("BTN_TL", 1):
                self._current_action.left_shoulder = True
            case ("BTN_TL", 0):
                self._current_action.left_shoulder = False
            # RB
            case ("BTN_TR", 1):
                self._current_action.right_shoulder = True
            case ("BTN_TR", 0):
                self._current_action.right_shoulder = False
            # left trigger
            case ("ABS_Z", v) if v > 127:  # noqa: PLR2004
                self._current_action.left_shoulder = True
            case ("ABS_Z", _):
                self._current_action.left_shoulder = False
            # right trigger
            case ("ABS_RZ", v) if v > 127:  # noqa: PLR2004
                self._current_action.right_shoulder = True
            case ("ABS_RZ", _):
                self._current_action.right_shoulder = False
            case _:
                pass

    def get_action(self) -> Action:
        return Action(**asdict(self._current_action))
