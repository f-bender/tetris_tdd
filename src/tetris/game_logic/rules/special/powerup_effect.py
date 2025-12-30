import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple, override

import numpy as np

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines
from tetris.game_logic.rules.board_manipulations.gravity import Gravity
from tetris.game_logic.rules.messages import (
    BlooperOverlayTrigger,
    BotAssistanceEnd,
    BotAssistanceStart,
    FillLinesEffectTrigger,
    FinishedLineFillMessage,
    GravityEffectTrigger,
    StartingLineFillMessage,
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
            BlooperEffect(),
        ]

    def reset(self) -> None:
        for effect in self._effects:
            effect.reset()

    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        for effect in self._effects:
            if effect.is_active:
                effect.apply_effect(frame_counter, action_counter, board, powerup_rule=self._powerup_rule)

    def trigger_random_effect(self, board: Board) -> None:
        inactive_effects = [effect for effect in self._effects if effect.is_available(board)]

        if inactive_effects:
            random.choice(inactive_effects).activate()


class PowerupEffect(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def is_available(self, board: Board) -> bool:
        return not self._active

    def reset(self) -> None:
        self._active = False

    def activate(self) -> None:
        self._active = True

    @abstractmethod
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None: ...


# --- Effects ---


class GravityEffect(PowerupEffect, Publisher, Subscriber):
    def __init__(self, per_col_probability: float = 1.0) -> None:
        super().__init__()

        self._per_col_probability = per_col_probability
        self._lines_being_currently_cleared: list[int] | None = None

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return publisher.game_index == self.game_index and isinstance(publisher, FillLines)

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case StartingLineFillMessage(filled_lines=lines_being_cleared):
                self._lines_being_currently_cleared = lines_being_cleared
            case FinishedLineFillMessage():
                self._lines_being_currently_cleared = None

    @override
    def is_available(self, board: Board) -> bool:
        # before testing, remove lines currently being cleared (if any)
        if self._lines_being_currently_cleared is not None:
            board = deepcopy(board)
            board.clear_lines(self._lines_being_currently_cleared)

        # if there is no "gravity bubbling" to be done, don't choose the gravity powerup effect
        # Note: in case of a "chain reaction", and a post-merge queue, gravity might end up actually not having an
        # effect by the time Gravity is applied, even though we would think so right now. But that should be rare.
        return Gravity.estimate_total_num_bubble_steps(board) != 0

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

    @override
    def is_available(self, board: Board) -> bool:
        # in case we have no subscribers, so this powerup has no effect
        # -> mark it as always unavailable (such that it is never selected to be triggered)
        # (note: this is the case when bot controller is used)
        return not self._active and len(self._subscribers) > 0

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


class FillLinesEffect(PowerupEffect, Publisher, Subscriber):
    def __init__(self, min_num_lines: int = 1, max_num_lines: int = 4) -> None:
        super().__init__()

        self._min_num_lines = min_num_lines
        self._max_num_lines = max_num_lines

        self._lines_being_currently_cleared: list[int] | None = None

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return publisher.game_index == self.game_index and isinstance(publisher, FillLines)

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case StartingLineFillMessage(filled_lines=lines_being_cleared):
                self._lines_being_currently_cleared = lines_being_cleared
            case FinishedLineFillMessage():
                self._lines_being_currently_cleared = None

    @override
    def is_available(self, board: Board) -> bool:
        non_empty_line_idxs = np.where(board.array_view_without_active_block().any(axis=1))[0]

        return len(non_empty_line_idxs) != 0 and (
            self._lines_being_currently_cleared is None
            or len(set(non_empty_line_idxs) - set(self._lines_being_currently_cleared)) != 0
        )

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        self.notify_subscribers(
            FillLinesEffectTrigger(num_lines=random.randint(self._min_num_lines, self._max_num_lines))
        )
        self._active = False


class SpawnPowerupsEffect(PowerupEffect, Subscriber):
    # limit the activation of this to once every 3 seconds (at 60 fps), to avoid a chain reaction resulting in many
    # powerups in a short amount of time
    _COOLDOWN_FRAMES = 180

    def __init__(self, min_num_powerups: int = 2, max_num_powerups: int = 4) -> None:
        super().__init__()

        self._min_num_powerups = min_num_powerups
        self._max_num_powerups = max_num_powerups

        self._cooldown_frame: int | None = None
        self._lines_being_currently_cleared: list[int] | None = None

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return publisher.game_index == self.game_index and isinstance(publisher, FillLines)

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case StartingLineFillMessage(filled_lines=lines_being_cleared):
                self._lines_being_currently_cleared = lines_being_cleared
            case FinishedLineFillMessage():
                self._lines_being_currently_cleared = None

    @override
    def is_available(self, board: Board) -> bool:
        if self._lines_being_currently_cleared is None:
            return bool(board.array_view_without_active_block().any())

        return bool(
            board.array_view_without_active_block()[
                list(set(range(board.height)) - set(self._lines_being_currently_cleared))
            ].any()
        )

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


class _BlooperTriggerMessage(NamedTuple):
    pass


class _BlooperCooldownMessage(NamedTuple):
    pass


class BlooperEffect(PowerupEffect, Publisher, Subscriber, Callback):
    # prevent activation while any blooper overlay is still present
    _COOLDOWN_FRAMES = 300

    def __init__(self) -> None:
        super().__init__()

        self._cooldown_frame: int | None = None
        self._on_cooldown = False
        self._game_over = False

    @override
    def is_available(self, board: Board) -> bool:
        return (
            not self._active
            and not self._on_cooldown
            # unavailable in case we have no subscribers (i.e. single-player, would have no effect)
            and len(self._subscribers) > 0
            # unavailable in case all other games are game over (same thing, would have no effect)
            and any(
                (isinstance(subscriber, BlooperEffect) and not subscriber.game_over) for subscriber in self._subscribers
            )
        )

    @override
    def on_game_over(self, game_index: int) -> None:
        if self._cooldown_frame is not None:
            # need to "release" the cooldown period - our apply_effect will not be called anymore after game over
            self.notify_subscribers(_BlooperCooldownMessage())

        self._active = False

        self._cooldown_frame = None
        self._on_cooldown = False
        self._game_over = True

    @override
    def on_game_start(self, game_index: int) -> None:
        self._game_over = False

    @property
    def game_over(self) -> bool:
        return self._game_over

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, BlooperEffect) and publisher.game_index != self.game_index

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case _BlooperTriggerMessage():
                self._on_cooldown = True
                self.notify_subscribers(BlooperOverlayTrigger())
            case _BlooperCooldownMessage():
                self._on_cooldown = False
            case _:
                pass

    @override
    def apply_effect(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, powerup_rule: "PowerupRule"
    ) -> None:
        if self._cooldown_frame is None:
            self.notify_subscribers(_BlooperTriggerMessage())
            self._cooldown_frame = frame_counter + self._COOLDOWN_FRAMES
        elif frame_counter >= self._cooldown_frame:
            self.notify_subscribers(_BlooperCooldownMessage())
            self._active = False
            self._cooldown_frame = None
