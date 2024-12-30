from collections.abc import Mapping
from typing import Self

import keyboard

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller

type Key = str | int


class KeyboardController(Controller):
    def __init__(self, action_to_keys: Mapping[str, list[Key]] | None = None) -> None:
        self._action_to_keys = action_to_keys or {
            "left": ["a", "left", "h"],
            "right": ["d", "right", "l"],
            "up": ["w", "up", "k"],
            "down": ["s", "down", "j"],
            "left_shoulder": ["q", "i"],
            "right_shoulder": ["e", "o"],
            "confirm": ["enter", "space", 82],  # numpad 0
            "cancel": ["esc", "ctrl"],
        }

        if (mapped_actions := set(self._action_to_keys.keys())) != (all_actions := set(Action._fields)):
            missing_actions = all_actions - mapped_actions
            invalid_actions = mapped_actions - all_actions

            message = "Error in provided `action_to_keys` map:"
            if missing_actions:
                message += f"\nMissing actions: {missing_actions}"
            if invalid_actions:
                message += f"\nInvalid actions: {missing_actions}"

            raise ValueError(message)

    @classmethod
    def wasd(cls) -> Self:
        return cls(
            action_to_keys={
                "left": ["a"],
                "right": ["d"],
                "up": ["w"],
                "down": ["s"],
                "left_shoulder": ["q"],
                "right_shoulder": ["e"],
                "confirm": ["space"],
                "cancel": ["esc"],
            }
        )

    @classmethod
    def arrow_keys(cls) -> Self:
        return cls(
            action_to_keys={
                "left": ["left"],
                "right": ["right"],
                "up": ["up"],
                "down": ["down"],
                "left_shoulder": [],
                "right_shoulder": [],
                "confirm": [82],  # numpad 0
                "cancel": ["ctrl"],
            }
        )

    @classmethod
    def vim(cls) -> Self:
        return cls(
            action_to_keys={
                "left": ["h"],
                "right": ["l"],
                "up": ["k"],
                "down": ["j"],
                "left_shoulder": ["i"],
                "right_shoulder": ["o"],
                "confirm": ["enter"],
                "cancel": ["backspace"],
            }
        )

    def get_action(self, board: Board | None = None) -> Action:
        return Action(
            **{
                action_name: any(keyboard.is_pressed(key) for key in key_list)
                for action_name, key_list in self._action_to_keys.items()
            },
        )
