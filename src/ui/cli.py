import atexit
import sys
from dataclasses import dataclass
from typing import Literal, NamedTuple, Self

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from ansi_extensions import color as colorx

# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package


@dataclass(frozen=True, slots=True)
class Vec:
    y: int
    x: int

    def __add__(self, other: "Vec") -> "Vec":
        return Vec(self.y + other.y, self.x + other.x)


class ColorPalette(NamedTuple):
    board_bg: str
    board_bg_alt: str
    board_fg: str

    @classmethod
    def from_rgb(
        cls,
        board_bg: tuple[int, int, int],
        board_bg_alt: tuple[int, int, int],
        board_fg: tuple[int, int, int],
        mode: Literal["palette", "truecolor"] = "truecolor",
    ) -> Self:
        color_fn = colorx.bg.rgb_palette if mode == "palette" else colorx.bg.rgb_truecolor
        return cls(
            board_bg=color_fn(*board_bg),
            board_bg_alt=color_fn(*board_bg_alt),
            board_fg=color_fn(*board_fg),
        )

    @classmethod
    def index_of_color(
        cls,
        color_name: Literal[
            "board_bg",
            "board_bg_alt",
            "board_fg",
        ],
    ) -> int:
        try:
            return cls._fields.index(color_name)
        except ValueError as e:
            raise ValueError(f"Invalid color name: '{color_name}'") from e


class CLI:
    PIXEL_WIDTH = 2  # how many terminal characters together form one pixel

    def __init__(
        self,
        color_palette: ColorPalette | None = None,
    ) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._color_palette = color_palette or ColorPalette.from_rgb(
            board_bg=(80, 80, 80), board_bg_alt=(60, 60, 60), board_fg=(200, 200, 200)
        )

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        # + 1 to make the interface 0-based (index of top CLI row, and left CLI column is actually 1, not 0)
        return cursor.goto(vec.y + 1, vec.x * CLI.PIXEL_WIDTH + 1)

    def initialize(self, board_height: int, board_width: int) -> None:
        self._board_background = np.full((board_height, board_width), ColorPalette.index_of_color("board_bg"))
        self._board_background[1::2, ::2] = ColorPalette.index_of_color("board_bg_alt")
        self._board_background[::2, 1::2] = ColorPalette.index_of_color("board_bg_alt")

        print(cursor.hide("") + cursor.erase(""), end="")
        atexit.register(self.terminate)

    def terminate(self) -> None:
        print(color.fx.reset + cursor.erase("") + self._cursor_goto(Vec(0, 0)) + cursor.show(""), end="")

    def draw(self, board: NDArray[np.bool]) -> None:
        assert self._board_background is not None, "draw() called before initialize()!"

        image_buffer = np.where(board, ColorPalette.index_of_color("board_fg"), self._board_background)

        if self._last_image_buffer is None:
            self._draw_array(Vec(0, 0), image_buffer)
        else:
            for y, x in zip(*(image_buffer != self._last_image_buffer).nonzero(), strict=True):
                self._draw_pixel(Vec(y, x), color_index=int(image_buffer[y, x]))

        sys.stdout.flush()  # make sure the changes are actually shown on screen

        self._last_image_buffer = image_buffer

    def game_over(self, board: NDArray[np.bool]) -> None:
        self.draw(board)
        print(self._cursor_goto(Vec(len(board), 0)) + color.fx.reset + "Game Over")

    def _draw_array(self, top_left: Vec, array: NDArray[np.uint8]) -> None:
        for idx, row in enumerate(array):
            self._draw_array_row(top_left=top_left + Vec(idx, 0), array_row=row)

    def _draw_array_row(self, top_left: Vec, array_row: NDArray[np.uint8]) -> None:
        print(
            CLI._cursor_goto(top_left)
            + "".join(self._color_palette[color_index] + " " * CLI.PIXEL_WIDTH for color_index in array_row),
            end="",
        )

    def _draw_pixel(self, position: Vec, color_index: int) -> None:
        print(self._cursor_goto(position) + self._color_palette[color_index] + " " * CLI.PIXEL_WIDTH, end="")
