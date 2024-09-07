import time

import pytest

from tetris.clock.amortizing import AmortizingClock
from tetris.clock.simple import SimpleClock
from tetris.game_logic.interfaces import Clock

TAKES_TOO_LONG_MESSAGE = "Takes too long. Enable when making changes to a clock class!"


def measure_average_time_between_ticks(
    clock: Clock, simulated_processing_time_between_ticks_s: float = 0, num_samples: int = 10
) -> float:
    clock.tick()
    before = time.perf_counter()

    for _ in range(num_samples):
        time.sleep(simulated_processing_time_between_ticks_s)
        clock.tick()

    after = time.perf_counter()

    return (after - before) / num_samples


@pytest.mark.skip(reason=TAKES_TOO_LONG_MESSAGE)
@pytest.mark.parametrize("clock_class", [SimpleClock, AmortizingClock])
def test_clock_tick_delay_without_processing_time(clock_class: type[SimpleClock | AmortizingClock]) -> None:
    fps = 60
    desired_tick_delay = 1 / fps
    clock = clock_class(fps=fps)

    tick_delay = measure_average_time_between_ticks(clock)

    assert tick_delay == pytest.approx(desired_tick_delay, abs=1e-3)


@pytest.mark.skip(reason=TAKES_TOO_LONG_MESSAGE)
@pytest.mark.parametrize("clock_class", [SimpleClock, AmortizingClock])
def test_clock_tick_delay_with_small_processing_time(clock_class: type[SimpleClock | AmortizingClock]) -> None:
    fps = 60
    desired_tick_delay = 1 / fps
    clock = clock_class(fps=fps)

    tick_delay = measure_average_time_between_ticks(
        clock, simulated_processing_time_between_ticks_s=desired_tick_delay / 2
    )

    assert tick_delay == pytest.approx(desired_tick_delay, abs=1e-3)


@pytest.mark.skip(reason=TAKES_TOO_LONG_MESSAGE)
@pytest.mark.parametrize("clock_class", [SimpleClock, AmortizingClock])
def test_clock_tick_delay_with_large_processing_time(clock_class: type[SimpleClock | AmortizingClock]) -> None:
    fps = 60
    desired_tick_delay = 1 / fps
    clock = clock_class(fps=fps)

    tick_delay = measure_average_time_between_ticks(
        clock, simulated_processing_time_between_ticks_s=desired_tick_delay * 1.5
    )

    assert tick_delay == pytest.approx(desired_tick_delay * 1.5, abs=1e-3)


@pytest.mark.skip(reason=TAKES_TOO_LONG_MESSAGE)
def test_amortizing_clock_is_accurate_over_time() -> None:
    fps = 60
    desired_tick_delay = 1 / fps

    clock = AmortizingClock(fps=fps, window_size=10)
    tick_delay = measure_average_time_between_ticks(clock, num_samples=100)
    assert tick_delay == pytest.approx(desired_tick_delay, abs=1e-4)

    clock = AmortizingClock(fps=fps, window_size=100)
    tick_delay = measure_average_time_between_ticks(clock, num_samples=100)
    assert tick_delay == pytest.approx(desired_tick_delay, abs=1e-5)
