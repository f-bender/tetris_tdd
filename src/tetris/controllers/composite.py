from collections.abc import Sequence
from typing import override

from tetris.game_logic.interfaces.controller import Action, Controller


class CompositeController(Controller):
    def __init__(self, controllers: Sequence[Controller]) -> None:
        super().__init__()

        self._controllers = controllers

    @property
    def symbol(self) -> str:
        return " + ".join(controller.symbol for controller in self._controllers)

    @override
    def get_action(self) -> Action:
        action = Action()

        for controller in self._controllers:
            action |= controller.get_action()

        return action
