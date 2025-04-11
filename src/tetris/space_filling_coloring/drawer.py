import random

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import color as colorx
from tetris.ansi_extensions import cursor as cursorx

last_drawn_array: NDArray[np.int32] | None = None
rng = random.Random()
draw_value: bool = False


def draw_array_fancy(array: NDArray[np.int32], *, rgb_range: tuple[int, int] = (50, 150)) -> None:
    global last_drawn_array  # noqa: PLW0603

    if last_drawn_array is None or last_drawn_array.shape != array.shape:
        print(cursor.erase(""), end="")
        print(cursor.goto(1, 1), end="")

        draw_full_array_raw(array, rgb_range=rgb_range)
    else:
        _draw_differences(array, last_drawn_array, rgb_range=rgb_range)

    # move cursor below the drawn array, such that prints outside this function show up there
    print(cursor.goto(array.shape[0] + 1) + cursorx.erase_to_end(""), end="", flush=True)

    last_drawn_array = array.copy()


def draw_full_array_raw(array: NDArray[np.int32], *, rgb_range: tuple[int, int] = (50, 150)) -> None:
    for row in array:
        print(
            "".join(
                _ansi_color_for(int(val), rgb_range=rgb_range) + (f"{val % 100:>2}" if draw_value else "  ")
                for val in row
            )
        )
    print(color.fx.reset, end="")


def _draw_differences(
    array: NDArray[np.int32], last_drawn_array: NDArray[np.int32], *, rgb_range: tuple[int, int]
) -> None:
    for y, x in np.argwhere(array != last_drawn_array):
        print(cursor.goto(y + 1, x * 2 + 1), end="")
        print(
            _ansi_color_for(val := int(array[y, x]), rgb_range=rgb_range)
            + (f"{val % 100:>2}" if draw_value else "  ")
            + color.fx.reset,
            end="",
        )


def _ansi_color_for(value: int, *, rgb_range: tuple[int, int]) -> str:
    global rng  # noqa: PLW0602

    if value == 0:
        return colorx.bg.rgb_truecolor(*[rgb_range[0] // 2] * 3)

    if value < 0:
        return colorx.bg.rgb_truecolor(*[(255 + rgb_range[1]) // 2] * 3)

    rng.seed(value)
    return colorx.bg.rgb_truecolor(*(rng.randrange(*rgb_range) for _ in range(3)))
