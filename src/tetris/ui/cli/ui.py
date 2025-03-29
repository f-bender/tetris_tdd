import atexit
import logging
import os
from math import ceil
from typing import TYPE_CHECKING, cast

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import cursor as cursorx
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.space_filling_coloring import concurrent_fill_and_colorize
from tetris.ui.cli.animations import Overlay
from tetris.ui.cli.buffered_printing import BufferedPrint
from tetris.ui.cli.color_palette import ColorPalette
from tetris.ui.cli.single_game_ui import Alignment, SingleGameUI, Text
from tetris.ui.cli.vec import Vec

if TYPE_CHECKING:
    from collections.abc import Generator


# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package

# NOTE: long-term, this should be split up into a general, reusable part, and a tetris-specific part

LOGGER = logging.getLogger(__name__)


class CLI(UI):
    PIXEL_WIDTH = 2  # how many terminal characters together form one pixel
    FRAME_WIDTH = 8  # width of the static frame around the board, in pixels

    EMOJI_THRESHOLD = 0x1000  # characters with unicodes higher than this are considered emojis

    MAX_BOARDS_SINGLE_ROW = 3

    def __init__(self, color_palette: ColorPalette | None = None, target_aspect_ratio: float = 16 / 9) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        # self._last_texts: set[Text] | None = None
        self._last_text_buffer: NDArray[np.str_] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._outer_background: NDArray[np.uint8] | None = None

        self._single_board_ui: SingleGameUI | None = None
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
            tetris_sparkle=(255, 255, 0),
            display_bg=(50, 50, 50),
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

        self._single_board_ui = SingleGameUI(self._create_board_background(board_height, board_width))
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

    def draw(self, elements: UiElements) -> None:
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
        text_buffer: NDArray[np.str_] = np.zeros_like(image_buffer, dtype=f"U{self.PIXEL_WIDTH}")
        all_overlays: list[Overlay] = []

        board_ui_height, board_ui_width = self._single_board_ui.total_size
        for single_ui_elements, offset in zip(elements.games, self._board_ui_offsets, strict=True):
            ui_array, ui_texts, overlays = self._single_board_ui.create_array_texts_animations(single_ui_elements)

            np.copyto(
                image_buffer[offset.y : offset.y + board_ui_height, offset.x : offset.x + board_ui_width],
                ui_array,
                where=self._single_board_ui.mask,
            )

            for text in ui_texts:
                text.position += offset
                self._add_text(text, text_buffer)

            for overlay in overlays:
                overlay.position += offset
            all_overlays.extend(overlays)

        # text_buffer.view(np.uint32)[(text_buffer.view(np.uint32) >= 0x1000)] = 0x00F6
        # LOGGER.debug(np.unique(text_buffer.view(np.uint32)))
        # LOGGER.debug(np.max(text_buffer.view(np.uint32)))
        # LOGGER.debug(
        #     np.where(np.logical_and((0x1F600 <= text_buffer.view(np.uint32)), (text_buffer.view(np.uint32) <= 0x1F64F)))
        # )
        # LOGGER.debug(f"\n{np.array2string(text_buffer.view(np.uint32), threshold=np.inf, max_line_width=np.inf)}")

        for overlay in all_overlays:
            self._draw_overlay(overlay=overlay, image_buffer=image_buffer, text_buffer=text_buffer)

        if self._last_image_buffer is None or self._last_text_buffer is None:
            self._draw_array(top_left=Vec(0, 0), array=image_buffer)
            for y, x in zip(*np.where(text_buffer), strict=True):
                self._draw_pixel(Vec(y, x), color_index=int(image_buffer[y, x]), text=cast("str", text_buffer[y, x]))
        else:
            for y, x in zip(
                *np.where((image_buffer != self._last_image_buffer) | (text_buffer != self._last_text_buffer)),
                strict=True,
            ):
                self._draw_pixel(Vec(y, x), color_index=int(image_buffer[y, x]), text=cast("str", text_buffer[y, x]))

        # if self._last_texts is None:
        #     self._draw_texts(texts)
        # else:
        #     removed_texts = self._last_texts - texts
        #     for text in removed_texts:
        #         text.text = " " * len(text.text)
        #     self._draw_texts(removed_texts)

        #     new_texts = texts - self._last_texts
        #     self._draw_texts(new_texts)

        self._last_image_buffer = image_buffer
        self._last_text_buffer = text_buffer
        # self._last_texts = texts

        self._buffered_print.print_and_restart_buffering()
        self._setup_cursor_for_normal_printing(image_height=len(image_buffer))

    @staticmethod
    def _add_text(text: Text, text_buffer: NDArray[np.str_]) -> None:
        min_characters_length = CLI._display_length(text.text)
        LOGGER.debug(f"{min_characters_length = }")

        if text.alignment is Alignment.RIGHT:
            character_offset = min_characters_length
        elif text.alignment is Alignment.CENTER:
            character_offset = ceil(min_characters_length / 2)
        else:
            character_offset = 0

        LOGGER.debug(f"{character_offset = }")
        pixel_offset = ceil(character_offset / CLI.PIXEL_WIDTH)
        LOGGER.debug(f"{pixel_offset = }")
        subpixel_offset = pixel_offset * CLI.PIXEL_WIDTH - character_offset
        LOGGER.debug(f"{subpixel_offset = }")
        LOGGER.debug(f"{text.text = }")
        characters = CLI._pixel_align_emojis(" " * subpixel_offset + text.text)
        LOGGER.debug(f"{characters = }")

        real_pixel_offset = pixel_offset + round((len(characters) - min_characters_length) / CLI.PIXEL_WIDTH)
        LOGGER.debug(f"{real_pixel_offset = }")
        position = text.position - Vec(0, real_pixel_offset)

        # TODO: check via ord(char) > 0x1000 if each character is an emoji
        # TODO: for each character, pixel-align it on the left side of a pixel, add spaces if necessary

        # if remainder := len(characters) % CLI.PIXEL_
        #     characters += " " * (CLI.PIXEL_WIDTH - remainder)

        # LOGGER.debug(f"{position.y = }, {position.x = }, {position.x + len(characters) // CLI.PIXEL_WIDTH = }")
        # LOGGER.debug(f"{characters = }, {len(characters) = }")
        LOGGER.debug(f"{text_buffer[position.y, position.x : position.x + len(characters) // CLI.PIXEL_WIDTH] = }")
        LOGGER.debug(f"{[characters[i : i + CLI.PIXEL_WIDTH] for i in range(0, len(characters), CLI.PIXEL_WIDTH)]}")
        text_buffer[position.y, position.x : position.x + ceil(len(characters) / CLI.PIXEL_WIDTH)] = [
            characters[i : i + CLI.PIXEL_WIDTH] for i in range(0, len(characters), CLI.PIXEL_WIDTH)
        ]

    @staticmethod
    def _pixel_align_emojis(text: str) -> str:
        result: list[str] = []

        for char in text:
            LOGGER.debug(f"{result = }")
            is_emoji = ord(char) > CLI.EMOJI_THRESHOLD
            if is_emoji and len(result) % CLI.PIXEL_WIDTH == CLI.PIXEL_WIDTH - 1:
                # if emoji is directly before a pixel border, insert a space to push it over the pixel border
                # otherwise, the character that will be written on the other side of the pixel border will write over
                # the emoji (even if that character is a space) (because the emoji takes up 2 spaces)
                result.append(" ")

            result.append(char)

            if is_emoji:
                # emoji takes up 2 spaces, so directly after emoji should be empty (will not be visible)
                result.append(" ")

        return "".join(result)

    @staticmethod
    def _display_length(text: str) -> int:
        return len(text) + sum(1 for char in text if ord(char) > CLI.EMOJI_THRESHOLD)

    @staticmethod
    def _draw_overlay(overlay: Overlay, image_buffer: NDArray[np.uint8], text_buffer: NDArray[np.str_]) -> None:
        # draw the overlay on the image
        np.copyto(
            image_buffer[
                overlay.position.y : overlay.position.y + overlay.height,
                overlay.position.x : overlay.position.x + overlay.width,
            ],
            overlay.frame,
            # using view() instead of astype() is an optimization that assumes that the int type of frame is
            # 8 bit wide!
            where=overlay.frame.view(bool),
        )

        # remove text where overlay is drawn (i.e. let overlay draw *over* the text)
        ys, xs = np.nonzero(overlay.frame)
        text_buffer[ys + overlay.position.y, xs + overlay.position.x] = ""

        # # alternative 1
        # text_buffer[
        #     overlay.position.y : overlay.position.y + overlay.height,
        #     overlay.position.x : overlay.position.x + overlay.width,
        # ][overlay.frame.view(bool)] = ""

        # # alternative 2
        # np.copyto(
        #     text_buffer[
        #         overlay.position.y : overlay.position.y + overlay.height,
        #         overlay.position.x : overlay.position.x + overlay.width,
        #     ],
        #     "",
        #     # using view() instead of astype() is an optimization that assumes that the int type of frame is
        #     # 8 bit wide!
        #     where=overlay.frame.view(bool),
        # )

    # def _draw_texts(self, texts: Iterable["Text"]) -> None:
    #     for text in texts:
    #         self._draw_text(text)

    # def _draw_text(self, text: "Text") -> None:
    #     if text.alignment is Alignment.RIGHT:
    #         character_offset = len(text.text)
    #     elif text.alignment is Alignment.CENTER:
    #         character_offset = ceil(len(text.text) / 2)
    #     else:
    #         character_offset = 0

    #     pixel_offset = ceil(character_offset / self.PIXEL_WIDTH)
    #     position = text.position - Vec(0, pixel_offset)
    #     characters = " " * (pixel_offset * self.PIXEL_WIDTH - character_offset) + text.text

    #     print(self._cursor_goto(position) + self._color_palette[text.bg_color_index] + characters, end="")

    def _handle_terminal_size_change(self) -> None:
        if (new_terminal_size := os.get_terminal_size()) != self._last_terminal_size:
            # terminal size changed: redraw everything
            self._last_image_buffer = None
            # self._last_texts = None
            self._last_text_buffer = None
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

    def _draw_pixel(self, position: Vec, color_index: int, text: str) -> None:
        print(
            self._cursor_goto(position)
            + self._color_palette[color_index]
            + ((text and text.ljust(CLI.PIXEL_WIDTH)) or " " * CLI.PIXEL_WIDTH),
            end="",
        )

    # def _draw_array(self, top_left: Vec, image_array: NDArray[np.uint8], text_array: NDArray[np.str_]) -> None:
    #     for idx, (image_row, text_row) in enumerate(zip(image_array, text_array, strict=True)):
    #         self._draw_array_row(top_left=top_left + Vec(idx, 0), array_row=image_row, text_row=text_row)

    # def _draw_array_row(self, top_left: Vec, array_row: NDArray[np.uint8], text_row: NDArray[np.str_]) -> None:
    #     print(
    #         self._cursor_goto(top_left)
    #         + "".join(
    #             self._color_palette[color_index] + cast("str", text).ljust(self.PIXEL_WIDTH)
    #             for color_index, text in zip(array_row, text_row, strict=True)
    #         ),
    #         end="",
    #     )
