import random
from abc import ABC, abstractmethod

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.messages import GravityEffectTrigger


class PowerupEffectManager:
    def __init__(self) -> None:
        self._effects: list[PowerupEffect] = [GravityEffect()]

    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        for effect in self._effects:
            if effect.is_active:
                effect.apply_effect(frame_counter, action_counter, board)

    def trigger_random_effect(self) -> None:
        inactive_effects = [effect for effect in self._effects if not effect.is_active]

        if inactive_effects:
            random.choice(inactive_effects).activate()


class PowerupEffect(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True

    @abstractmethod
    def apply_effect(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None: ...


class GravityEffect(PowerupEffect, Publisher):
    def apply_effect(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        self.notify_subscribers(GravityEffectTrigger(per_col_probability=1))
        self._active = False
