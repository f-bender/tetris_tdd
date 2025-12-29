import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.messages import (
    BotAssistanceEnd,
    BotAssistanceStart,
    FillLinesEffectTrigger,
    GravityEffectTrigger,
)

if TYPE_CHECKING:
    from tetris.game_logic.rules.special.powerup import PowerupRule


class PowerupEffectManager:
    def __init__(self, powerup_rule: "PowerupRule") -> None:
        self._powerup_rule = powerup_rule

        self._effects: list[PowerupEffect] = [
            GravityEffect(),
            BotAssistanceEffect(),
            FillLinesEffect(),
            SpawnPowerupsEffect(),
        ]

    def reset(self) -> None:
        for effect in self._effects:
            effect.reset()

    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        for effect in self._effects:
            if effect.is_active:
                effect.apply_effect(frame_counter, action_counter, board, powerup_rule=self._powerup_rule)

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

    def reset(self) -> None:
        self._active = False

    def activate(self) -> None:
        self._active = True

    @abstractmethod
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None: ...


# --- Effects ---


class GravityEffect(PowerupEffect, Publisher):
    def __init__(self, per_col_probability: float = 1.0) -> None:
        super().__init__()

        self._per_col_probability = per_col_probability

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        self.notify_subscribers(GravityEffectTrigger(per_col_probability=self._per_col_probability))
        self._active = False


class BotAssistanceEffect(PowerupEffect, Publisher):
    def __init__(
        self,
        min_effect_duration_frames: int = 300,
        max_effect_duration_frames: int = 600,
    ) -> None:
        super().__init__()

        self._min_effect_duration_frames = min_effect_duration_frames
        self._max_effect_duration_frames = max_effect_duration_frames

        self._end_frame: int | None = None

    @property
    @override
    def is_active(self) -> bool:
        # in case we have no subscribers, so this powerup has no effect
        # -> mark it as always active (such that it is never selected to be triggered)
        # (note: this is the case when bot controller is used)
        return self._active or not self._subscribers

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        if self._end_frame is None:
            self._end_frame = frame_counter + random.randint(
                self._min_effect_duration_frames, self._max_effect_duration_frames
            )
            self.notify_subscribers(BotAssistanceStart())
        elif frame_counter >= self._end_frame:
            self.notify_subscribers(BotAssistanceEnd())
            self._active = False
            self._end_frame = None


class FillLinesEffect(PowerupEffect, Publisher):
    def __init__(self, min_num_lines: int = 1, max_num_lines: int = 4) -> None:
        super().__init__()

        self._min_num_lines = min_num_lines
        self._max_num_lines = max_num_lines

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        self.notify_subscribers(
            FillLinesEffectTrigger(num_lines=random.randint(self._min_num_lines, self._max_num_lines))
        )
        self._active = False


class SpawnPowerupsEffect(PowerupEffect):
    # limit the activation of this to once per second (at 60 fps), to avoid a chain reaction resulting in many powerups
    # in a short amount of time
    _COOLDOWN_FRAMES = 60

    def __init__(self, min_num_powerups: int = 2, max_num_powerups: int = 4) -> None:
        super().__init__()

        self._min_num_powerups = min_num_powerups
        self._max_num_powerups = max_num_powerups

        self._cooldown_frame: int | None = None

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        if self._cooldown_frame is None:
            powerup_rule.spawn_powerups(
                board.array_view_without_active_block(), random.randint(self._min_num_powerups, self._max_num_powerups)
            )
            self._cooldown_frame = frame_counter + self._COOLDOWN_FRAMES
        elif frame_counter >= self._cooldown_frame:
            self._active = False
            self._cooldown_frame = None
