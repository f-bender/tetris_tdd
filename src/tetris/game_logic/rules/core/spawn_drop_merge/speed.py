from collections.abc import Iterator, Mapping
from math import ceil, floor
from types import MappingProxyType
from typing import NamedTuple, Protocol, Self

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.messages import FinishedLineClearMessage, NewLevelMessage


class IntIterWithFloatAverage(Iterator[int]):
    """An iterator that yields integers, just above or below the specified target average float.

    The average of the values yielded by this iterator will converge towards the target average.
    """

    def __init__(self, target_average: float) -> None:
        self._target_average = target_average
        self.reset()

    def reset(self) -> None:
        self._num_values = 0
        self._current_average: float | None = None

    @property
    def target_average(self) -> float:
        return self._target_average

    @target_average.setter
    def target_average(self, target_average: float) -> None:
        self._target_average = target_average
        self.reset()

    def __next__(self) -> int:
        overshot = (self._current_average or 0) > self._target_average
        next_value = floor(self._target_average) if overshot else ceil(self._target_average)

        self._current_average = ((self._current_average or 0) * self._num_values + next_value) / (self._num_values + 1)
        self._num_values += 1

        return next_value

    def __iter__(self) -> Self:
        return self


class SpeedStrategy(Protocol):
    def should_trigger(self, frames_since_last_drop: int, *, quick_drop_held: bool) -> bool: ...


class SpeedStrategyImpl(Callback):
    """Strategy for whether to drop/merge the current block, given the number of frames since it was spawned."""

    def __init__(self, base_interval: float = 48, quick_interval_factor: float = 8) -> None:
        """Initialize the SpeedStrategy.

        Needs to be registered as a callback to the game in order to work correctly.

        Args:
            base_interval: Initial value for the number of frames between drops while the quick-drop-action is *not*
                held. The default value of 30 is fine-tuned for 60 FPS gameplay. If changed by a subclass, that interval
                will be changed to this base value again whenever a new game starts.
            quick_interval_factor: `normal_interval` is divided by this factor to obtain the number of frames between
                drops while the quick-drop-action *is* held.
        """
        super().__init__()

        self._base_interval = base_interval
        self._quick_interval_factor = quick_interval_factor

        self._normal_interval_iter = IntIterWithFloatAverage(self._base_interval)
        self._quick_interval_iter = IntIterWithFloatAverage(max(self._base_interval / self._quick_interval_factor, 1))

        self.set_interval(base_interval)

    @property
    def normal_interval(self) -> float:
        return self._normal_interval

    def set_interval(self, normal_interval: float) -> None:
        self._normal_interval = normal_interval
        self._quick_interval = max(normal_interval / self._quick_interval_factor, 1)

        self._normal_interval_iter.target_average = self._normal_interval
        self._quick_interval_iter.target_average = self._quick_interval

        self._next_normal_interval_int = next(self._normal_interval_iter)
        self._next_quick_interval_int = next(self._quick_interval_iter)

    def on_game_start(self) -> None:
        self.set_interval(self._base_interval)

    def should_trigger(self, frames_since_last_drop: int, *, quick_drop_held: bool) -> bool:
        if quick_drop_held:
            if should_trigger := frames_since_last_drop >= self._next_quick_interval_int:
                self._next_quick_interval_int = next(self._quick_interval_iter)

            return should_trigger

        if should_trigger := frames_since_last_drop >= self._next_normal_interval_int:
            self._next_normal_interval_int = next(self._normal_interval_iter)

        return should_trigger


class LineClearSpeedUp(Subscriber, Callback):
    """Decorator for the SpeedStrategyImpl class, which induces a speedup every N line clears."""

    def __init__(
        self,
        speed_strategy_impl: SpeedStrategyImpl | None = None,
        *,
        minimum_normal_interval: float = 4,
        speedup_factor: float = 1.1,
        line_clears_between_speedup: int = 10,
    ) -> None:
        """Initialize the LineClearSpeedupStrategy.

        Args:
            speed_strategy_impl: SpeedStrategyImpl instance being decorated.
            minimum_normal_interval: Minimum value for the number of frames between drops while the quick-drop-action is
                *not* held. The default value of 4 is fine-tuned for 60 FPS gameplay.
            speedup_factor: Factor by which the normal interval is divided to obtain the new interval after a speedup.
            line_clears_between_speedup: Number of line clears after which the speedup is triggered.
        """
        super().__init__()

        self._speed_strategy_impl = speed_strategy_impl or SpeedStrategyImpl()

        self._cleared_lines = 0
        self._line_clears_between_speedup = line_clears_between_speedup

        self._minimum_normal_interval = minimum_normal_interval
        self._speedup_factor = speedup_factor

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, ClearFullLines) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if (
            not isinstance(message, FinishedLineClearMessage)
            or self._speed_strategy_impl.normal_interval <= self._minimum_normal_interval
        ):
            return

        self._cleared_lines += len(message.cleared_lines)

        while self._cleared_lines >= self._line_clears_between_speedup:
            self._speed_strategy_impl.set_interval(
                max(self._speed_strategy_impl.normal_interval / self._speedup_factor, self._minimum_normal_interval)
            )
            self._cleared_lines -= self._line_clears_between_speedup

    def on_game_start(self) -> None:
        self._cleared_lines = 0

    def should_trigger(self, frames_since_last_drop: int, *, quick_drop_held: bool) -> bool:
        return self._speed_strategy_impl.should_trigger(
            frames_since_last_drop=frames_since_last_drop, quick_drop_held=quick_drop_held
        )


class LevelSpeedUp(Subscriber, Callback):
    """Decorator for the SpeedStrategyImpl class, which induces a speedup every level.

    Its default values mirror the NES Tetris behavior, taken from https://tetris.fandom.com/wiki/Tetris_(NES,_Nintendo).
    """

    _DEFAULT_INTERVAL_BY_LEVEL = MappingProxyType(
        {
            0: 48,
            1: 43,
            2: 38,
            3: 33,
            4: 28,
            5: 23,
            6: 18,
            7: 13,
            8: 8,
            9: 6,
            10: 5,
            13: 5,
            16: 3,
            19: 2,
            29: 1,
        }
    )

    def __init__(
        self,
        speed_strategy_impl: SpeedStrategyImpl | None = None,
        interval_by_level: Mapping[int, float] = _DEFAULT_INTERVAL_BY_LEVEL,
    ) -> None:
        """Initialize the LevelSpeedUp strategy.

        Args:
            speed_strategy_impl: SpeedStrategyImpl instance being decorated.
            interval_by_level: Mapping from level number to the number of frames between drops while the
                quick-drop-action is *not* held. The default values mirror the NES Tetris behavior. Levels not present
                in this mapping will not trigger a speed change when reached.
        """
        super().__init__()

        if 0 not in interval_by_level:
            msg = "LevelSpeedUp requires interval_by_level to define an initial interval for level 0."
            raise ValueError(msg)

        self._interval_by_level = interval_by_level
        self._speed_strategy_impl = speed_strategy_impl or SpeedStrategyImpl(base_interval=self._interval_by_level[0])

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, LevelTracker) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, NewLevelMessage) and message.level in self._interval_by_level:
            self._speed_strategy_impl.set_interval(self._interval_by_level[message.level])

    def should_trigger(self, frames_since_last_drop: int, *, quick_drop_held: bool) -> bool:
        return self._speed_strategy_impl.should_trigger(
            frames_since_last_drop=frames_since_last_drop, quick_drop_held=quick_drop_held
        )
