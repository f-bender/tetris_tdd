import atexit
from enum import IntEnum, auto
from functools import lru_cache
from itertools import cycle, islice
from numbers import Number
from time import perf_counter
from typing import Callable, Literal, NamedTuple, Protocol, cast

import numpy as np
from ansi import color, cursor
from ansi.colour.rgb import rgb256
from ansi_extensions import cursor as cursorx
import ansi_extensions
from game_logic.components import Board
from numpy.typing import NDArray
from PIL import ImageGrab
from ansi_extensions import color as colorx

# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package

# TODO:
# - overwrite only the character that have changed by moving the cursor to these positions (ANSI sequences)
#   - for this purpose, probably change the interface to the such that is actually just passes the boolean numpy array
#     which is simpler and more performant to operate on (finding the differences)
# - [might not play well together with the change above - delay and maybe cancel this altogether] don't set the
#   background for every single block, but set it once and keep it until there is a character which needs another color

# TODO: try handing this file to ChatGPT-4o / Claude 3.5, explain the interfaces, and how the UI should look like


class RGB(NamedTuple):
    r: int
    g: int
    b: int


class Vec(NamedTuple):
    y: int
    x: int

    def __add__(self, other: "Vec") -> "Vec":
        return Vec(self.y + other.y, self.x + other.x)


class Color(IntEnum):
    BOARD_BG = auto()
    BOARD_BG_ALT = auto()
    BOARD_FG = auto()


COLOR_PALETTE: dict[int, RGB] = {
    Color.BOARD_BG: RGB(80, 80, 80),
    Color.BOARD_BG_ALT: RGB(60, 60, 60),
    Color.BOARD_FG: RGB(200, 200, 200),
}

# TODO figure out why the first row is screwed up!
# TODO understand why the "+ 1" is necessary in the _cursor_goto method

class CLI:
    PIXEL_WIDTH = 2  # 2 terminal characters form one pixel
    BG_COLOR_FN: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor
    FG_COLOR_FN: Callable[[int, int, int], str] = colorx.fg.rgb_truecolor

    LIGHT_ACTIVE_CELL = color.bg.truecolor(255, 255, 255)("  ")
    DARK_ACTIVE_CELL = color.bg.truecolor(int(255 * 0.9), int(255 * 0.9), int(255 * 0.9))("  ")
    LIGHT_BACKGROUND = color.bg.truecolor(80, 80, 80)("  ")
    DARK_BACKGROUND = color.bg.truecolor(int(80 * 0.9), int(80 * 0.9), int(80 * 0.9))("  ")

    def __init__(self) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._board_background: NDArray[np.uint8] | None = None

    @classmethod
    def set_color_mode(cls, mode: Literal["palette", "truecolor"]) -> None:
        if mode == "palette":
            cls.BG_COLOR_FN = colorx.bg.rgb_palette
            cls.FG_COLOR_FN = colorx.fg.rgb_palette
        elif mode == "truecolor":
            cls.BG_COLOR_FN = colorx.bg.rgb_truecolor
            cls.FG_COLOR_FN = colorx.fg.rgb_truecolor
        else:
            raise ValueError(f"Unknown color mode: {mode}")

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        return cursor.goto(vec.y, vec.x * CLI.PIXEL_WIDTH + 1)

    @staticmethod
    def _get_color(color: Color | RGB) -> RGB:
        if isinstance(color, Number):
            return COLOR_PALETTE[color]
        return color

    def initialize(self, board_height: int, board_width: int) -> None:
        self._board_background = np.full((board_height, board_width), Color.BOARD_BG)
        self._board_background[1::2, ::2] = Color.BOARD_BG_ALT
        self._board_background[::2, 1::2] = Color.BOARD_BG_ALT

        print(cursor.hide("") + cursor.erase(""), end="")
        # atexit.register(self.terminate)

    def terminate(self) -> None:
        print(cursor.show("") + cursor.erase("") + color.fx.reset, end="")

    def draw(self, board: NDArray[np.bool]) -> None:
        assert self._board_background is not None, "draw() called before initialize()!"
        # t0 = perf_counter()
        # array = np.vectorize(lambda x: np.array([80, 80, 80]) if x else np.array([200, 200, 200]), signature="()->(n)")(
        #     board
        # )
        # vectorize = perf_counter() - t0

        t0 = perf_counter()
        image_buffer = np.where(board, Color.BOARD_FG, self._board_background)
        where = perf_counter() - t0

        # t0 = perf_counter()
        # self._draw_array(top_left=Vec(0, 0), array=image_buffer)
        # draw = perf_counter() - t0

        # print(color.fx.reset)
        # print(f"{vectorize=}\n{draw=}")
        # print(f"{where=}\n{draw=}")
        # self._draw_rectangle(top_left=Vec(0, 0), bottom_right=Vec(*board.shape), color=RGB(80, 80, 80))

        # for (y, x), active in np.ndenumerate(board):
        #     if active:
        #         self._draw_pixel(Vec(y, x), color=RGB(200, 200, 200))

        if self._last_image_buffer is not None:
            for y, x in zip(*(image_buffer != self._last_image_buffer).nonzero()):
                self._draw_pixel(Vec(y, x), color=COLOR_PALETTE[cast(int, image_buffer[y, x])])
        else:
            self._draw_array(Vec(0, 0), image_buffer)

        self._last_image_buffer = image_buffer

    def game_over(self, board: NDArray[np.bool]) -> None:
        self.draw(board)
        print(color.fx.reset)
        print("Game Over")

    @staticmethod
    def _draw_board_background(top_left: Vec, shape: Vec) -> None:
        CLI._draw_rectangle(
            top_left=top_left,
            bottom_right=top_left + shape,
            color=RGB(80, 80, 80),
            alt_color=RGB(60, 60, 60),
        )

    @staticmethod
    def _draw_clipboard_image(top_left: Vec = Vec(0, 0)) -> None:
        image = np.array(ImageGrab.grabclipboard(), dtype=np.uint8)[..., :3]
        CLI._draw_array(top_left, image)

    @staticmethod
    def _draw_array(top_left: Vec, array: NDArray[np.uint8]) -> None:
        for idx, row in enumerate(array):
            CLI._draw_array_row(top_left=top_left + Vec(idx, 0), array_row=row)

    @staticmethod
    def _draw_array_row(top_left: Vec, array_row: NDArray[np.uint8]) -> None:
        print(
            CLI._cursor_goto(top_left)
            + "".join(CLI.BG_COLOR_FN(*CLI._get_color(color)) + " " * CLI.PIXEL_WIDTH for color in array_row),
            end="",
        )

    @staticmethod
    def _draw_rectangle(top_left: Vec, bottom_right: Vec, color: RGB, alt_color: RGB | None = None) -> None:
        for idx, y in enumerate(range(top_left[0], bottom_right[0])):
            if idx % 2 == 1 and alt_color is not None:
                CLI._draw_horizontal_line(y=y, x_range=(top_left[1], bottom_right[1]), color=alt_color, alt_color=color)
            else:
                CLI._draw_horizontal_line(y=y, x_range=(top_left[1], bottom_right[1]), color=color, alt_color=alt_color)

    @staticmethod
    def _draw_rectangle_outline(top_left: Vec, bottom_right: Vec, color: RGB, alt_color: RGB | None) -> None:
        pass

    @staticmethod
    def _draw_horizontal_line(y: int, x_range: tuple[int, int], color: RGB, alt_color: RGB | None) -> None:
        print(CLI._cursor_goto(Vec(y, x_range[0])), end="")
        length = x_range[1] - x_range[0]

        if not alt_color:
            print(CLI.BG_COLOR_FN(*color) + " " * CLI.PIXEL_WIDTH * length, end="")
            return

        print(
            "".join(
                islice(
                    cycle(
                        [
                            CLI.BG_COLOR_FN(*color) + " " * CLI.PIXEL_WIDTH,
                            CLI.BG_COLOR_FN(*alt_color) + " " * CLI.PIXEL_WIDTH,
                        ]
                    ),
                    length,
                )
            ),
            end="",
        )

    @staticmethod
    def _draw_vertical_line(y_range: tuple[int, int], x: int, color: RGB, alt_color: RGB | None = None) -> None:
        pass

    @staticmethod
    def _draw_pixel(position: Vec, color: RGB) -> None:
        print(CLI._cursor_goto(position) + CLI.BG_COLOR_FN(*color) + " " * CLI.PIXEL_WIDTH, end="")
