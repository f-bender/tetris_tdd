from collections import deque
import time

from game_logic.interfaces import Clock


# NOTE: one potentially undesirable property: after lag (processing time > desired tick delay), up to the next
# `window_size` ticks may happen "instantly"
class AmortizingClock(Clock):
    def __init__(self, fps: float = 60, window_size: int = 60) -> None:
        self._tick_delay = 1 / fps
        self._last_ticks: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        if (
            self._last_ticks
            and (
                remaining_delay := self._last_ticks[0] + self._tick_delay * len(self._last_ticks) - time.perf_counter()
            )
            > 0
        ):
            time.sleep(remaining_delay)

        self._last_ticks.append(time.perf_counter())
