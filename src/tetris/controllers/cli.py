"""DEPRECATED!

This is a neat zero-dependency concept, but it's too clunky for a game that works based on a fixed framerate. It's only
really useful when the application waits and performs actions based and timed completely on the user's input.
"""
# mypy: ignore-errors

import os
import sys
import warnings
from functools import partial
from threading import Thread
from time import perf_counter
from typing import NoReturn

from tetris.game_logic.interfaces.controller import Action, Controller

if os.name == "nt":
    import msvcrt

    ctrl_c = 3

    def get_char() -> str:
        char = msvcrt.getwch()
        if ord(char) == ctrl_c:
            raise KeyboardInterrupt
        return char
else:
    import tty

    tty.setcbreak(sys.stdin)
    get_char = partial(sys.stdin.read, 1)


class CliButtonListener(Thread):
    INITIAL_HOLD_MAX_DELAY_S = 0.525
    CONTINUOUS_HOLD_MAX_DELAY_S = 0.04

    def __init__(self) -> None:
        super().__init__()
        self._last_pressed_char: str | None = None
        self._continuous_hold: bool = False
        self._last_press_time: float | None = None

    def run(self) -> NoReturn:
        while True:
            pressed_char = get_char()
            now = perf_counter()

            if pressed_char == self._last_pressed_char:
                assert self._last_press_time is not None
                if not self._continuous_hold and (now - self._last_press_time < self.INITIAL_HOLD_MAX_DELAY_S):
                    self._continuous_hold = True
                elif self._continuous_hold and (now - self._last_press_time > self.CONTINUOUS_HOLD_MAX_DELAY_S):
                    self._continuous_hold = False
            else:
                self._continuous_hold = False

            self._last_pressed_char = pressed_char
            self._last_press_time = now

    def get_held_char(self) -> str | None:
        if not self._last_pressed_char or not self._last_press_time:
            return None

        since_last_delay_s = perf_counter() - self._last_press_time

        if since_last_delay_s < self.CONTINUOUS_HOLD_MAX_DELAY_S or (
            not self._continuous_hold and since_last_delay_s < self.INITIAL_HOLD_MAX_DELAY_S
        ):
            return self._last_pressed_char

        return None


class CliController(Controller):
    """A primitive controller working with only the Python standard library.

    It's unable to recognize multiple button presses (and thus perform multiple actions) at once.
    It also requires a separate thread to function, and can not distinguish between fast pressing and holding down.
    """

    def __init__(self) -> None:
        warnings.warn(
            "CliController is deprecated: Is too clunky and doesn't work well for a clock-driven setting "
            "(as opposed to a user-input-driven setting)",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self._button_listener = CliButtonListener()
        self._button_listener.start()

    def get_action(self) -> Action:
        match self._button_listener.get_held_char():
            case "a":
                return Action(left=True)
            case "s":
                return Action(down=True)
            case "d":
                return Action(right=True)
            case "q":
                return Action(left_shoulder=True)
            case "e":
                return Action(right_shoulder=True)
            case _:
                return Action()
