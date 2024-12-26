import time


class SimpleClock:
    def __init__(self, fps: float = 60) -> None:
        self._tick_delay = 1 / fps
        self._last_tick: float | None = None

    def tick(self) -> None:
        if (
            self._last_tick is not None
            and (remaining_delay := self._last_tick + self._tick_delay - time.perf_counter()) > 0
        ):
            time.sleep(remaining_delay)

        self._last_tick = time.perf_counter()

    def overdue(self) -> bool:
        return self._last_tick is not None and self._last_tick + self._tick_delay < time.perf_counter()

    def reset(self) -> None:
        self._last_tick = None
