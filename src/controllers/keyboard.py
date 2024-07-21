from typing import Mapping

import keyboard
from game_logic.interfaces.controller import Action, Controller

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

    def get_action(self) -> Action:
        return Action(
            **{
                action_name: any(keyboard.is_pressed(key) for key in key_list)
                for action_name, key_list in self._action_to_keys.items()
            }
        )
