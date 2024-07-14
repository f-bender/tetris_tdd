from typing import Mapping
import keyboard
from game_logic.interfaces.controller import Action, Controller


class KeyboardController(Controller):
    def __init__(self, action_to_keys: Mapping[str, list[str]] | None = None) -> None:
        self._action_to_keys = action_to_keys or {
            "move_left": ["a", "left", "h"],
            "move_right": ["d", "right", "l"],
            "rotate_left": ["q"],
            "rotate_right": ["e", "w", "up", "k"],
            "quick_drop": ["s", "down", "j"],
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
