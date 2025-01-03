from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.rule import Subscriber
from tetris.rules.core.clear_full_lines_rule import LineClearMessage


class SpeedStrategy(Callback):
    """Strategy for whether to drop/merge the current block, given the number of frames since it was spawned."""

    def __init__(self, base_interval: int = 25, quick_interval_factor: float = 8) -> None:
        """Initialize the SpeedStrategy.

        Needs to be registered as a callback to the game in order to work correctly.

        Args:
            base_interval: Initial value for the number of frames between drops while the quick-drop-action is *not*
                held. The default value of 25 is fine-tuned for 60 FPS gameplay. If changed by a subclass, that interval
                will be changed to this base value again whenever a new game starts.
            quick_interval_factor: `normal_interval` is divided by this factor to obtain the number of frames between
                drops while the quick-drop-action *is* held.
        """
        self._base_interval = base_interval
        self._quick_interval_factor = quick_interval_factor

        self._set_interval(base_interval)

    def _set_interval(self, normal_interval: int) -> None:
        self._normal_interval = normal_interval
        self._quick_interval = max(round(normal_interval / self._quick_interval_factor), 1)

    def on_game_start(self) -> None:
        self._set_interval(self._base_interval)

    def should_trigger(self, frames_since_last_drop: int, *, quick_drop_held: bool) -> bool:
        if quick_drop_held:
            return frames_since_last_drop >= self._quick_interval

        return frames_since_last_drop >= self._normal_interval


class LineClearSpeedupStrategy(SpeedStrategy, Subscriber):
    """Speed strategy that speeds up the block drop interval after a certain number of line clears.

    Needs to be registered as a subscriber to the ClearFullLinesRule, and as a callback to the Game in order to work
    correctly!
    """

    def __init__(
        self,
        base_interval: int = 25,
        quick_interval_factor: float = 8,
        *,
        minimum_normal_interval: int = 8,
        speedup_factor: float = 1.1,
        line_clears_between_speedup: int = 10,
    ) -> None:
        """Initialize the LineClearSpeedupStrategy.

        Args:
            base_interval: Initial value for the number of frames between drops while the quick-drop-action is *not*
                held. The default value of 25 is fine-tuned for 60 FPS gameplay. If changed by a subclass, that interval
                will be changed to this base value again whenever a new game starts.
            quick_interval_factor: `normal_interval` is divided by this factor to obtain the number of frames between
                drops while the quick-drop-action *is* held.
            minimum_normal_interval: Minimum value for the number of frames between drops while the quick-drop-action is
                *not* held. The default value of 8 is fine-tuned for 60 FPS gameplay.
            speedup_factor: Factor by which the normal interval is divided to obtain the new interval after a speedup.
            line_clears_between_speedup: Number of line clears after which the speedup is triggered.
        """
        super().__init__(base_interval=base_interval, quick_interval_factor=quick_interval_factor)

        self._cleared_lines = 0
        self._line_clears_between_speedup = line_clears_between_speedup

        self._minimum_normal_interval = minimum_normal_interval
        self._speedup_factor = speedup_factor
        self._float_interval: float = self._normal_interval

    def notify(self, message: NamedTuple) -> None:
        if not isinstance(message, LineClearMessage) or self._normal_interval <= self._minimum_normal_interval:
            return

        self._cleared_lines += len(message.cleared_lines)

        if self._cleared_lines >= self._line_clears_between_speedup:
            self._float_interval = self._normal_interval / self._speedup_factor
            self._set_interval(max(round(self._float_interval), self._minimum_normal_interval))

            self._cleared_lines -= self._line_clears_between_speedup

    def on_game_start(self) -> None:
        super().on_game_start()

        self._cleared_lines = 0
        self._float_interval = self._normal_interval
