import time
from collections import deque


# NOTE: one potentially undesirable property: after lag (processing time > desired tick delay), up to the next
# `window_size` ticks may happen "instantly"
class AmortizingClock:
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
            time.sleep(min(remaining_delay, self._tick_delay))

        self._last_ticks.append(time.perf_counter())

    def overdue(self) -> bool:
        return (
            bool(self._last_ticks)
            and self._last_ticks[0] + self._tick_delay * len(self._last_ticks) < time.perf_counter()
        )

    def reset(self) -> None:
        self._last_ticks.clear()
