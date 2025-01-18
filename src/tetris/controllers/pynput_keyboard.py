import logging
from collections.abc import Mapping
from typing import Self, cast

from pynput import keyboard

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller

LOGGER = logging.getLogger(__name__)


class PynputKeyboardController(Controller):
    def __init__(self, action_to_keys: Mapping[str, list[str | int]] | None = None) -> None:
        """Initialize the PynputKeyboardController.

        Args:
            action_to_keys: Mapping from action names (see class Action) to lists of key representations for the keys
                that all should be able to trigger the respective action.
                Key representations can be int (VK key codes) or str (key chars like "a", or key names like "enter".)

        Raises:
            ValueError: If not all actions are covered by action_to_keys, or it contains an invalid action name.
        """
        action_to_keys = action_to_keys or {
            "left": ["a", "left", "h"],
            "right": ["d", "right", "l"],
            "up": ["w", "up", "k"],
            "down": ["s", "down", "j"],
            "left_shoulder": ["q", "i"],
            "right_shoulder": ["e", "o"],
            "confirm": [
                "enter",
                "space",
                96,  # vk: numpad 0
            ],
            "cancel": ["esc", "ctrl"],
        }

        if (mapped_actions := set(action_to_keys.keys())) != (all_actions := set(Action._fields)):
            missing_actions = all_actions - mapped_actions
            invalid_actions = mapped_actions - all_actions

            message = "Error in provided `action_to_keys` map:"
            if missing_actions:
                message += f"\nMissing actions: {missing_actions}"
            if invalid_actions:
                message += f"\nInvalid actions: {missing_actions}"

            raise ValueError(message)

        # map key names/chars to their VKs
        self._action_to_keys: dict[str, list[int | str]] = {
            action: [
                self._key_str_to_vk_or_char(key_repr) if isinstance(key_repr, str) else key_repr
                for key_repr in key_reprs
            ]
            for action, key_reprs in action_to_keys.items()
        }

        # VKs of currently pressed keys
        self._pressed_key_vks: set[int] = set()
        # chars of currently pressed keys
        self._pressed_key_chars: set[str] = set()

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    @staticmethod
    def _key_str_to_vk_or_char(key_str: str) -> int | str:
        if key := getattr(keyboard.Key, key_str, None):
            return key.value.vk

        # if key_str is not a special key (like space, enter, ...) we assume it's already the canonical char (like "a")
        return key_str

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key and (key_code := self._parse_key(key)) is not None:
            if key_code.vk is not None:
                self._pressed_key_vks.add(key_code.vk)

            if key_code.char is not None:
                self._pressed_key_chars.add(key_code.char)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key and (key_code := self._parse_key(key)) is not None:
            if key_code.vk is not None:
                self._pressed_key_vks.discard(key_code.vk)

            if key_code.char is not None:
                self._pressed_key_chars.discard(key_code.char)

    def _parse_key(self, key: keyboard.Key | keyboard.KeyCode) -> keyboard.KeyCode:
        key = self._listener.canonical(key)

        if isinstance(key, keyboard.Key):
            key = cast(keyboard.KeyCode, key.value)

        return key

    def __del__(self) -> None:
        """Clean up the keyboard listener when the object is destroyed."""
        if hasattr(self, "_listener"):
            self._listener.stop()

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
                "confirm": [96],  # vk: numpad 0
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
                action_name: any(
                    (
                        (isinstance(key, int) and key in self._pressed_key_vks)
                        or (isinstance(key, str) and key in self._pressed_key_chars)
                    )
                    for key in key_list
                )
                for action_name, key_list in self._action_to_keys.items()
            }
        )
