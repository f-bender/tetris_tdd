import atexit
from dataclasses import dataclass

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import cursor as cursorx
from tetris.ui.cli.buffered_printing import BufferedPrint
from tetris.ui.cli.color_palette import ColorPalette

# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package

# NOTE: long-term, this should be split up into a general, reusable part, and a tetris-specific part


@dataclass(frozen=True, slots=True)
class Vec:
    y: int
    x: int

    def __add__(self, other: "Vec") -> "Vec":
        return Vec(self.y + other.y, self.x + other.x)


class CLI:
    PIXEL_WIDTH = 2  # how many terminal characters together form one pixel

    def __init__(self, color_palette: ColorPalette | None = None) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._color_palette = color_palette or ColorPalette.from_rgb(
            board_bg=(80, 80, 80), board_bg_alt=(60, 60, 60), board_fg=(200, 200, 200)
        )
        self._buffered_print = BufferedPrint()

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        # + 1 to make the interface 0-based (index of top CLI row, and left CLI column is actually 1, not 0)
        return cursor.goto(vec.y + 1, vec.x * CLI.PIXEL_WIDTH + 1)

    def initialize(self, board_height: int, board_width: int) -> None:
        self._initialized_board_background(board_height, board_width)
        self._initialize_terminal(board_height)

    def _initialized_board_background(self, board_height: int, board_width: int) -> None:
        self._board_background = np.full((board_height, board_width), ColorPalette.index_of_color("board_bg"))
        self._board_background[1::2, ::2] = ColorPalette.index_of_color("board_bg_alt")
        self._board_background[::2, 1::2] = ColorPalette.index_of_color("board_bg_alt")

    def _initialize_terminal(self, board_height: int) -> None:
        atexit.register(self.terminate)
        self._initialize_cursor()
        self._setup_cursor_for_normal_printing(board_height)
        self._buffered_print.start_buffering()

    def _initialize_cursor(self) -> None:
        print(cursor.hide("") + cursor.erase(""), end="")

    def terminate(self) -> None:
        if self._buffered_print.is_active():
            self._buffered_print.discard_and_reset_buffer()
        print(color.fx.reset + cursor.erase("") + self._cursor_goto(Vec(0, 0)) + cursor.show(""), end="")

    def draw(self, board: NDArray[np.bool]) -> None:
        assert (
            self._board_background is not None
        ), "background not initialized, likely draw() was called before initialize()!"
        assert (
            self._buffered_print.is_active()
        ), "buffered printing not active, likely draw() called before initialize()!"

        image_buffer = np.where(board, ColorPalette.index_of_color("board_fg"), self._board_background)

        if self._last_image_buffer is None:
            self._draw_array(Vec(0, 0), image_buffer)
        else:
            for y, x in zip(*(image_buffer != self._last_image_buffer).nonzero(), strict=True):
                self._draw_pixel(Vec(y, x), color_index=int(image_buffer[y, x]))

        self._last_image_buffer = image_buffer

        self._buffered_print.print_and_restart_buffering()
        self._setup_cursor_for_normal_printing(board_height=len(board))

    def _setup_cursor_for_normal_printing(self, board_height: int) -> None:
        """Move the cursor below the board, clear any colors, and erase anything below the board.

        This allows usage of print outside the class in a well-defined manner, appearing below the board and being reset
        every after every call to this function.
        """
        print(self._cursor_goto(Vec(board_height, 0)) + color.fx.reset + cursorx.erase_to_end(""))

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
