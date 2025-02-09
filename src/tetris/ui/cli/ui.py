import atexit
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Self

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import cursor as cursorx
from tetris.game_logic.interfaces.ui import UI
from tetris.space_filling_coloring import concurrent_fill_and_colorize
from tetris.ui.cli.buffered_printing import BufferedPrint
from tetris.ui.cli.color_palette import ColorPalette

if TYPE_CHECKING:
    from collections.abc import Generator

# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package

# NOTE: long-term, this should be split up into a general, reusable part, and a tetris-specific part

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Vec:
    y: int
    x: int

    def __add__(self, other: Self) -> "Vec":
        return Vec(self.y + other.y, self.x + other.x)


class CLI(UI):
    PIXEL_WIDTH = 2  # how many terminal characters together form one pixel
    FRAME_WIDTH = 8  # width of the static frame around the board, in pixels

    MAX_BOARDS_SINGLE_ROW = 3

    def __init__(self, color_palette: ColorPalette | None = None, target_aspect_ratio: float = 16 / 9) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._outer_background: NDArray[np.uint8] | None = None

        self._single_board_ui: _SingleBoardUI | None = None
        self._board_ui_offsets: list[Vec] | None = None

        self._color_palette = color_palette or ColorPalette.from_rgb(
            **{f"outer_bg_progress_{i}": (127 + 10 * (i - 5),) * 3 for i in range(1, 11)},  # type: ignore[arg-type]
            outer_bg_1=(46, 0, 2),
            outer_bg_2=(39, 85, 10),
            outer_bg_3=(123, 1, 6),
            outer_bg_4=(15, 33, 4),
            board_bg=(50, 50, 50),
            board_bg_alt=(30, 30, 30),
            block_1=(160, 1, 241),
            block_2=(248, 230, 8),
            block_3=(0, 255, 255),
            block_4=(239, 130, 1),
            block_5=(2, 241, 2),
            block_6=(51, 153, 255),
            block_7=(240, 0, 1),
            block_neutral=(200, 200, 200),
        )
        self._buffered_print = BufferedPrint()
        self._startup_animation_iter: (
            Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]
            | None
        ) = None
        self._startup_finished: bool = False

        self._target_aspect_ratio = target_aspect_ratio

        self._last_terminal_size = os.get_terminal_size()

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        # + 1 to make the interface 0-based (index of top CLI row, and left CLI column is actually 1, not 0)
        return cursor.goto(vec.y + 1, vec.x * CLI.PIXEL_WIDTH + 1)

    def initialize(self, board_height: int, board_width: int, num_boards: int) -> None:
        if num_boards <= 0:
            msg = "num_boards must be greater than 0"
            raise ValueError(msg)

        self._single_board_ui = _SingleBoardUI(self._create_board_background(board_height, board_width))
        self._initialize_board_ui_offsets(num_boards)
        self._initialize_terminal()

    def _create_board_background(self, board_height: int, board_width: int) -> NDArray[np.uint8]:
        board_background = np.full((board_height, board_width), ColorPalette.index_of_color("board_bg"), dtype=np.uint8)

        board_background[1::2, ::2] = ColorPalette.index_of_color("board_bg_alt")
        board_background[::2, 1::2] = ColorPalette.index_of_color("board_bg_alt")

        return board_background

    def _initialize_board_ui_offsets(self, num_boards: int) -> None:
        assert self._single_board_ui is not None

        board_ui_height, board_ui_width = self._single_board_ui.total_size

        num_rows, num_cols = self._compute_num_rows_cols(num_boards)

        self._board_ui_offsets = [
            Vec(
                self.FRAME_WIDTH + y * (board_ui_height + self.FRAME_WIDTH),
                self.FRAME_WIDTH + x * (board_ui_width + self.FRAME_WIDTH),
            )
            for y in range(num_rows)
            for x in range(num_cols)
            if y * num_cols + x < num_boards
        ]

    def _create_outer_background_mask(self) -> NDArray[np.bool]:
        if self._board_ui_offsets is None or self._single_board_ui is None:
            msg = "board UI not initialized, likely initialize() was not called before advance_startup()!"
            raise RuntimeError(msg)

        board_ui_height, board_ui_width = self._single_board_ui.total_size
        total_height = max(offset.y for offset in self._board_ui_offsets) + board_ui_height + self.FRAME_WIDTH
        total_width = max(offset.x for offset in self._board_ui_offsets) + board_ui_width + self.FRAME_WIDTH

        outer_background_mask = np.ones((total_height, total_width), dtype=np.bool)

        # cut out the board UIs from the outer background
        for offset in self._board_ui_offsets:
            outer_background_mask[
                offset.y : offset.y + board_ui_height,
                offset.x : offset.x + board_ui_width,
            ] = ~self._single_board_ui.mask

        # make sure the size of the space to be filled is divisible by 4 (otherwise it can't be filed with tetrominos)
        match int(np.sum(outer_background_mask) % 4):
            case 1:
                outer_background_mask[-1, -1] = False
            case 2:
                outer_background_mask[-1, -1] = False
                outer_background_mask[0, 0] = False
            case 3:
                outer_background_mask[-1, -1] = False
                outer_background_mask[-1, -2] = False
                outer_background_mask[-2, -1] = False

        return outer_background_mask

    def _compute_num_rows_cols(self, num_boards: int) -> tuple[int, int]:
        assert self._single_board_ui is not None

        board_height, board_width = self._single_board_ui.board_size
        height_added_per_board = board_height + self.FRAME_WIDTH
        width_added_per_board = board_width + self.FRAME_WIDTH

        # solution from WolframAlpha, prompted with
        # `(c * w + f) / (r * h + f) = a, r * c = n, solve for r, c`
        # where c: num_cols, w: width_added_per_board, f: frame_width, r: num_rows, h: height_added_per_board,
        #       a: target_aspect_ratio, n: num_boards
        # (`ceil` added myself to make it integers)
        num_cols = ceil(
            (
                (
                    (self._target_aspect_ratio - 1) ** 2 * self.FRAME_WIDTH**2
                    + 4 * self._target_aspect_ratio * height_added_per_board * num_boards * width_added_per_board
                )
                ** 0.5
                + (self._target_aspect_ratio - 1) * self.FRAME_WIDTH
            )
            / (2 * width_added_per_board)
        )
        num_rows = ceil(num_boards / num_cols)

        # reduce the number of columns, as long as this doesn't increase the required number of rows
        while num_cols > 1 and ceil(num_boards / (num_cols - 1)) == num_rows:
            num_cols -= 1

        return num_rows, num_cols

    def _initialize_terminal(self) -> None:
        atexit.register(self.terminate)
        self._initialize_cursor()
        self._buffered_print.start_buffering()

    def _initialize_cursor(self) -> None:
        print(cursor.hide("") + cursor.erase(""), end="")

    def terminate(self) -> None:
        if self._buffered_print.is_active():
            self._buffered_print.discard_and_reset_buffer()
        print(color.fx.reset + cursor.erase("") + self._cursor_goto(Vec(0, 0)) + cursor.show(""), end="")

    def advance_startup(self) -> bool:
        """Advance the startup animation by one step. Returns True if the animation is finished."""
        if self._startup_finished:
            return True

        self._ensure_startup_animation_iter_initialized()
        assert self._startup_animation_iter is not None

        try:
            filled_space, colored_space = next(self._startup_animation_iter)
        except StopIteration as e:
            filled_space, colored_space = e.value
            self._startup_finished = True
            # let the startup animation objects be garbage collected
            self._startup_animation_iter = None

        self._outer_background = self._background_from_filled_colored(filled_space, colored_space)

        return self._startup_finished

    def _ensure_startup_animation_iter_initialized(self) -> None:
        if self._startup_animation_iter is not None:
            return

        outer_background_mask = self._create_outer_background_mask()

        self._startup_animation_iter = concurrent_fill_and_colorize.fill_and_colorize(
            outer_background_mask, minimum_separation_steps=15
        )

    def _background_from_filled_colored(
        self, filled_space: NDArray[np.int32], colored_space: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        return np.where(
            colored_space > 0,
            colored_space + ColorPalette.outer_bg_index_offset() - 1,
            np.where(
                filled_space > 0,
                filled_space % 10 + ColorPalette.outer_bg_progress_index_offset(),
                self._color_palette.index_of_color("empty"),
            ),
        )

    def draw(self, boards: Iterable[NDArray[np.uint8]]) -> None:
        if self._single_board_ui is None or self._board_ui_offsets is None:
            msg = "board UI not initialized, likely draw() was called before initialize()!"
            raise RuntimeError(msg)
        if not self._buffered_print.is_active():
            msg = "buffered printing not active, likely draw() was called before initialize()!"
            raise RuntimeError(msg)

        if self._outer_background is None:
            LOGGER.warning(
                "Outer background not initialized, likely draw() was called before advance_startup()! "
                "Using an empty background."
            )
            self._outer_background = self._create_outer_background_mask().astype(np.uint8)

        self._handle_terminal_size_change()

        image_buffer = self._outer_background.copy()

        board_ui_height, board_ui_width = self._single_board_ui.total_size
        for board, offset in zip(boards, self._board_ui_offsets, strict=True):
            if board is None:
                continue
            image_buffer[offset.y : offset.y + board_ui_height, offset.x : offset.x + board_ui_width] = (
                self._single_board_ui.create_as_array(board)
            )

        if self._last_image_buffer is None:
            self._draw_array(Vec(0, 0), image_buffer)
        else:
            for y, x in zip(*(image_buffer != self._last_image_buffer).nonzero(), strict=True):
                self._draw_pixel(Vec(y, x), color_index=int(image_buffer[y, x]))

        self._last_image_buffer = image_buffer

        self._buffered_print.print_and_restart_buffering()
        self._setup_cursor_for_normal_printing(image_height=len(image_buffer))

    def _handle_terminal_size_change(self) -> None:
        if (new_terminal_size := os.get_terminal_size()) != self._last_terminal_size:
            # terminal size changed: redraw everything
            self._last_image_buffer = None
            self._last_terminal_size = new_terminal_size
            print(cursor.erase(""), end="")

    def _setup_cursor_for_normal_printing(self, image_height: int) -> None:
        """Move the cursor below the board, clear any colors, and erase anything below the board.

        This allows usage of print outside the class in a well-defined manner, appearing below the board and being reset
        every after every call to this function.
        """
        print(self._cursor_goto(Vec(image_height, 0)) + color.fx.reset + cursorx.erase_to_end(""))

    def _draw_array(self, top_left: Vec, array: NDArray[np.uint8]) -> None:
        for idx, row in enumerate(array):
            self._draw_array_row(top_left=top_left + Vec(idx, 0), array_row=row)

    def _draw_array_row(self, top_left: Vec, array_row: NDArray[np.uint8]) -> None:
        print(
            self._cursor_goto(top_left)
            + "".join(self._color_palette[color_index] + " " * CLI.PIXEL_WIDTH for color_index in array_row),
            end="",
        )

    def _draw_pixel(self, position: Vec, color_index: int) -> None:
        print(self._cursor_goto(position) + self._color_palette[color_index] + " " * CLI.PIXEL_WIDTH, end="")


@dataclass(frozen=True, slots=True)
class _SingleBoardUI:
    board_background: NDArray[np.uint8]

    @property
    def board_size(self) -> tuple[int, int]:
        return tuple(self.board_background.shape)

    @property
    def total_size(self) -> tuple[int, int]:
        return self.board_size

    @property
    def mask(self) -> NDArray[np.bool]:
        return np.ones_like(self.board_background, dtype=np.bool)

    def create_as_array(self, board: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return np.where(board, board + ColorPalette.block_color_index_offset() - 1, self.board_background)
