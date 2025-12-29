import atexit
import logging
import os
from dataclasses import astuple, dataclass
from functools import cached_property, lru_cache
from math import ceil
from typing import TYPE_CHECKING, cast

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import cursor as cursorx
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.space_filling_coloring import fill_and_colorize
from tetris.ui.cli.buffered_printing import BufferedPrint
from tetris.ui.cli.color_palette import BackgroundColorType, ColorPalette
from tetris.ui.cli.dynamic_layer import random_layer
from tetris.ui.cli.single_game_ui import Alignment, Overlay, SingleGameUI, Text
from tetris.ui.cli.vec import Vec

if TYPE_CHECKING:
    from collections.abc import Generator


# see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 for a list of ANSI escape codes,
# NOT ALL of which are provided in the `ansi` package

# NOTE: long-term, this should be split up into a general, reusable part, and a tetris-specific part

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DynamicLayerConfig:
    wavelength_factor: float = 0.35
    propagation_speed_factor: float = 3
    layer_n_mixed: int = 3
    dynamic_background_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.wavelength_factor <= 0 or self.propagation_speed_factor <= 0:
            msg = "wavelength_factor and propagation_speed_factor have to be > 0"
            raise ValueError(msg)

        if self.layer_n_mixed < 1:
            msg = "layer_n_mixed has to be >= 1"
            raise ValueError(msg)


_DEFAULT_DYNAMIC_BACKGROUND_CONFIG = DynamicLayerConfig()


class CLI(UI):
    _PIXEL_WIDTH = 2  # how many terminal characters together form one pixel

    _EMOJI_THRESHOLD = 0x2500  # characters with unicodes higher than this are considered emojis

    # if there are more pixel changes than this in one row, the whole row is re-drawn (not just the changed pixels)
    # (this is roughly the threshold where whole-row draws become more efficient)
    _MAX_PIXELS_PER_ROW_TO_DELTA_DRAW = 10

    def __init__(  # noqa: PLR0913
        self,
        *,
        color_palette: ColorPalette | None = None,
        frame_width: int = 8,  # width of the static frame around and between the single games' UIs, in pixels
        target_aspect_ratio: float = 16 / 9,
        randomize_background_colors_on_levelup: bool = True,
        dynamic_background_config: DynamicLayerConfig = _DEFAULT_DYNAMIC_BACKGROUND_CONFIG,
        text_fx: str = str(color.fx.bold),
    ) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._last_text_buffer: NDArray[np.str_] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._outer_background: NDArray[np.uint8] | None = None

        self._single_game_ui: SingleGameUI | None = None
        self._game_ui_offsets: list[Vec] | None = None

        self._color_palette = color_palette or ColorPalette.default()

        self._dynamic_background_config = dynamic_background_config

        bg_color_type = BackgroundColorType.random(
            dynamic_probability=self._dynamic_background_config.dynamic_background_probability
        )
        self._dynamic_background_during_startup = bg_color_type is BackgroundColorType.DYNAMIC
        if not self._dynamic_background_during_startup:
            self._color_palette.randomize_outer_bg_colors(
                bg_color_type=bg_color_type,
                dynamic_background_speed_factor=self._dynamic_background_config.propagation_speed_factor,
            )

        self._buffered_print = BufferedPrint()
        self._startup_animation_iter: (
            Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]
            | None
        ) = None
        self._startup_finished: bool = False

        self._frame_width = frame_width
        self._target_aspect_ratio = target_aspect_ratio
        self._text_fx = text_fx

        self._last_terminal_size = os.get_terminal_size()

        self._randomize_background_colors_on_levelup = randomize_background_colors_on_levelup
        self._level: int | None = None
        self._dynamic_layer: NDArray[np.uint32] | None = None

    @cached_property
    def total_size(self) -> tuple[int, int]:
        if self._game_ui_offsets is None or self._single_game_ui is None:
            msg = "board UI not initialized, cannot get total_size before initialize() is called!"
            raise RuntimeError(msg)

        board_ui_height, board_ui_width = self._single_game_ui.total_size
        total_height = max(offset.y for offset in self._game_ui_offsets) + board_ui_height + self._frame_width
        total_width = max(offset.x for offset in self._game_ui_offsets) + board_ui_width + self._frame_width

        return total_height, total_width

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        # + 1 to make the interface 0-based (index of top CLI row, and left CLI column is actually 1, not 0)
        return cursor.goto(vec.y + 1, vec.x * CLI._PIXEL_WIDTH + 1)

    def initialize(self, board_height: int, board_width: int, num_boards: int) -> None:
        if num_boards <= 0:
            msg = "num_boards must be greater than 0"
            raise ValueError(msg)

        self._single_game_ui = SingleGameUI(board_height, board_width, self._PIXEL_WIDTH, total_num_games=num_boards)
        self._initialize_board_ui_offsets(num_boards)
        self._initialize_terminal()

        self._dynamic_layer = self._randomized_dynamic_layer()

    def _randomized_dynamic_layer(self) -> NDArray[np.uint32]:
        return (
            random_layer(
                size=self.total_size,
                n_mixed_primitive_layers=self._dynamic_background_config.layer_n_mixed,
            )
            / self._dynamic_background_config.wavelength_factor
        ).astype(np.uint32)

    def _initialize_board_ui_offsets(self, num_games: int) -> None:
        assert self._single_game_ui is not None

        game_ui_height, game_ui_width = self._single_game_ui.total_size

        num_rows, num_cols = self._compute_num_rows_cols(num_games)

        self._game_ui_offsets = [
            Vec(
                self._frame_width + y * (game_ui_height + self._frame_width),
                self._frame_width + x * (game_ui_width + self._frame_width),
            )
            for y in range(num_rows)
            for x in range(num_cols)
            if y * num_cols + x < num_games
        ]

    def _create_outer_background_mask(self) -> NDArray[np.bool]:
        if self._game_ui_offsets is None or self._single_game_ui is None:
            msg = "board UI not initialized, likely initialize() was not called before advance_startup()!"
            raise RuntimeError(msg)

        outer_background_mask = np.ones(self.total_size, dtype=np.bool)

        # cut out the board UIs from the outer background
        for offset in self._game_ui_offsets:
            outer_background_mask[
                offset.y : offset.y + self._single_game_ui.total_size[0],
                offset.x : offset.x + self._single_game_ui.total_size[1],
            ] = ~self._single_game_ui.mask

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
            case _:
                pass

        assert np.sum(outer_background_mask) % 4 == 0

        return outer_background_mask

    def _compute_num_rows_cols(self, num_games: int) -> tuple[int, int]:
        assert self._single_game_ui is not None

        single_game_ui_height, single_game_ui_width = self._single_game_ui.total_size
        height_added_per_game = single_game_ui_height + self._frame_width
        width_added_per_game = single_game_ui_width + self._frame_width

        # solution from WolframAlpha, prompted with
        # `(c * w + f) / (r * h + f) = a, r * c = n, solve for r, c`
        # where c: num_cols, w: width_added_per_game, f: frame_width, r: num_rows, h: height_added_per_game,
        #       a: target_aspect_ratio, n: num_games
        # (`ceil` added myself to make it integers)
        num_cols = ceil(
            (
                (
                    (self._target_aspect_ratio - 1) ** 2 * self._frame_width**2
                    + 4 * self._target_aspect_ratio * height_added_per_game * num_games * width_added_per_game
                )
                ** 0.5
                + (self._target_aspect_ratio - 1) * self._frame_width
            )
            / (2 * width_added_per_game)
        )
        num_rows = ceil(num_games / num_cols)

        # reduce the number of columns, as long as this doesn't increase the required number of rows
        while num_cols > 1 and ceil(num_games / (num_cols - 1)) == num_rows:
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

        self._startup_animation_iter = fill_and_colorize.fill_and_colorize_concurrently(
            outer_background_mask, minimum_separation_steps=15
        )

    def _background_from_filled_colored(
        self, filled_space: NDArray[np.int32], colored_space: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        return np.where(
            colored_space > 0,
            colored_space
            - 1
            + (
                ColorPalette.DYNAMIC_BACKGROUND_INDEX_0
                if self._dynamic_background_during_startup
                else ColorPalette.outer_bg_index_offset()
            ),
            np.where(
                filled_space > 0,
                filled_space % 10 + ColorPalette.outer_bg_progress_index_offset(),
                self._color_palette.index_of_color("empty"),
            ),
        ).astype(np.uint8)

    def draw(self, elements: UiElements) -> None:
        if self._single_game_ui is None or self._game_ui_offsets is None or self._dynamic_layer is None:
            msg = "game UI not initialized, likely draw() was called before initialize()!"
            raise RuntimeError(msg)
        if not self._buffered_print.is_active():
            msg = "buffered printing not active, likely draw() was called before initialize()!"
            raise RuntimeError(msg)

        if self._outer_background is None:
            LOGGER.warning(
                "Outer background not initialized, likely draw() was called before advance_startup()! "
                "Using an empty background."
            )
            self._outer_background = np.full(self.total_size, ColorPalette.index_of_color("empty"))

        # NOTE: this *could* technically overflow, but it would have to run for 2 years straight (at 60 fps),
        # and even then it wouldn't break but just look weird for a few seconds
        self._dynamic_layer += 1

        self._handle_terminal_size_change()
        self._handle_level_change(elements)

        image_buffer = self._outer_background.copy()
        text_buffer: NDArray[np.str_] = np.zeros_like(image_buffer, dtype=f"U{self._PIXEL_WIDTH}")

        self._draw_to_buffers(elements=elements, image_buffer=image_buffer, text_buffer=text_buffer)

        self._draw_buffers_to_screen(image_buffer=image_buffer, text_buffer=text_buffer)

        self._last_image_buffer = image_buffer
        self._last_text_buffer = text_buffer

        self._buffered_print.print_and_restart_buffering()
        self._setup_cursor_for_normal_printing(image_height=len(image_buffer))

    def _draw_buffers_to_screen(self, image_buffer: NDArray[np.uint8], text_buffer: NDArray[np.str_]) -> None:
        if self._last_image_buffer is not None:
            # we have a last buffer: only draw where it changed
            # note: dynamic background colors change all the time even though their index entry in the image buffer
            # doesn't
            changed_mask = (image_buffer != self._last_image_buffer) | (
                image_buffer >= ColorPalette.DYNAMIC_BACKGROUND_INDEX_0
            )

            if self._last_text_buffer is not None:
                changed_mask |= text_buffer != self._last_text_buffer
            else:
                # we have no last text buffer: consider *every* text a change
                changed_mask = np.logical_or(changed_mask, text_buffer)
        else:
            # we have no last buffer: consider *everything* changed
            # (rare path, so performance is not that critical)
            changed_mask = np.ones_like(self._outer_background, dtype=np.bool)

        for y, row in enumerate(changed_mask):
            xs = np.nonzero(row)[0]
            if xs.size == 0:
                continue

            if len(xs) <= self._MAX_PIXELS_PER_ROW_TO_DELTA_DRAW:
                for x in xs:
                    self._draw_pixel(
                        Vec(y, x), color_index=int(image_buffer[y, x]), text=cast("str", text_buffer[y, x])
                    )
            else:
                # note: drawing whole rows with text in them breaks the UI when emojis are used in the text, so first
                # draw the entire row, then over-draw the pixels with text on them
                self._draw_array_row(top_left=Vec(y, 0), array_row=image_buffer[y])
                # note: its important that texts are drawn quickly after background at that location, to avoid
                # flickering (noticeable amount of time with just the background without text)
                # So drawing the text here, *immediately* after the row (and not later in bulk) is actually very much
                # intended, and not just a side-effect of the design
                for x in np.nonzero(text_buffer[y])[0]:
                    self._draw_pixel(
                        Vec(y, x), color_index=int(image_buffer[y, x]), text=cast("str", text_buffer[y, x])
                    )

    def _draw_to_buffers(
        self, elements: UiElements, image_buffer: NDArray[np.uint8], text_buffer: NDArray[np.str_]
    ) -> None:
        assert self._single_game_ui is not None
        assert self._game_ui_offsets is not None

        all_overlays: list[Overlay] = []

        single_game_ui_height, single_game_ui_width = self._single_game_ui.total_size
        for single_ui_elements, offset in zip(elements.games, self._game_ui_offsets, strict=True):
            ui_array, ui_texts, overlays = self._single_game_ui.create_array_texts_animations(single_ui_elements)

            np.copyto(
                image_buffer[offset.y : offset.y + single_game_ui_height, offset.x : offset.x + single_game_ui_width],
                ui_array,
                where=self._single_game_ui.mask,
            )

            for text in ui_texts:
                text.position += offset
                self._add_text(text=text, text_buffer=text_buffer)

            for overlay in overlays:
                overlay.position += offset
            all_overlays.extend(overlays)

        for overlay in all_overlays:
            self._draw_overlay(overlay=overlay, image_buffer=image_buffer, text_buffer=text_buffer)

    def _handle_level_change(self, elements: UiElements) -> None:
        if not self._randomize_background_colors_on_levelup:
            return

        combined_level = sum(game.level for game in elements.games)
        if self._level is not None and combined_level != self._level:
            self._randomize_outer_bg_palette()
            self._last_image_buffer = None  # force full redraw on next draw call

        self._level = combined_level

    def _randomize_outer_bg_palette(self) -> None:
        assert self._outer_background is not None

        dynamic_background_before = np.any(self._outer_background >= ColorPalette.DYNAMIC_BACKGROUND_INDEX_0)
        bg_color_type = BackgroundColorType.random(
            dynamic_probability=self._dynamic_background_config.dynamic_background_probability
        )

        if bg_color_type is not BackgroundColorType.DYNAMIC and dynamic_background_before:
            # reset from dynamic background, in case we don't want it now and had it before
            self._outer_background[self._outer_background != ColorPalette.index_of_color("empty")] -= (
                ColorPalette.STATIC_DYNAMIC_IDX_OFFSET
            )
        elif bg_color_type is BackgroundColorType.DYNAMIC and not dynamic_background_before:
            # set dynamic background, in case we want it now but didn't before
            self._outer_background[self._outer_background != ColorPalette.index_of_color("empty")] += (
                ColorPalette.STATIC_DYNAMIC_IDX_OFFSET
            )

        if bg_color_type is BackgroundColorType.DYNAMIC:
            self._dynamic_layer = self._randomized_dynamic_layer()

        self._color_palette.randomize_outer_bg_colors(bg_color_type=bg_color_type)

    @staticmethod
    def _add_text(text: Text, text_buffer: NDArray[np.str_]) -> None:
        characters, offset = CLI._processed_text_and_offset(text=text.text, alignment=text.alignment)
        position = text.position - Vec(0, offset)

        text_buffer[position.y, position.x : position.x + ceil(len(characters) / CLI._PIXEL_WIDTH)] = [
            characters[i : i + CLI._PIXEL_WIDTH] for i in range(0, len(characters), CLI._PIXEL_WIDTH)
        ]

    @lru_cache
    @staticmethod
    def _processed_text_and_offset(text: str, alignment: Alignment) -> tuple[str, int]:
        # NOTE: this code *does* work with pixel widths other than 2, but looks the best with pixel width = 2
        min_characters_length = CLI._display_length(text)

        if alignment is Alignment.RIGHT:
            character_offset = min_characters_length
        elif alignment is Alignment.CENTER:
            character_offset = ceil(min_characters_length / 2)
        else:
            character_offset = 0

        pixel_offset = ceil(character_offset / CLI._PIXEL_WIDTH)
        subpixel_offset = pixel_offset * CLI._PIXEL_WIDTH - character_offset
        characters = CLI._pixel_align_emojis(" " * subpixel_offset + text)

        # in case emoji-pixel-alignment has increased the length of `characters`, increase the pixel offset (only full
        # pixel steps at this point)
        if alignment is Alignment.RIGHT:
            real_pixel_offset = pixel_offset + round(
                (len(characters) - (min_characters_length + subpixel_offset)) / CLI._PIXEL_WIDTH
            )
        elif alignment is Alignment.CENTER:
            real_pixel_offset = pixel_offset + round(
                (len(characters) - (min_characters_length + subpixel_offset)) / CLI._PIXEL_WIDTH / 2
            )
        else:
            real_pixel_offset = pixel_offset

        return characters, real_pixel_offset

    @staticmethod
    def _pixel_align_emojis(text: str) -> str:
        result: list[str] = []

        for char in text:
            is_emoji = ord(char) > CLI._EMOJI_THRESHOLD
            if is_emoji and len(result) % CLI._PIXEL_WIDTH == CLI._PIXEL_WIDTH - 1:
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
        # emojis are displayed with a width of 2
        return sum(2 if ord(char) > CLI._EMOJI_THRESHOLD else 1 for char in text)

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

        # and add texts that belong to the overlay *on top* of it
        for text in overlay.texts:
            text.position += overlay.position
            CLI._add_text(text=text, text_buffer=text_buffer)

    def _handle_terminal_size_change(self) -> None:
        if (new_terminal_size := os.get_terminal_size()) != self._last_terminal_size:
            # terminal size changed: redraw everything
            self._last_image_buffer = None
            self._last_text_buffer = None
            self._last_terminal_size = new_terminal_size
            print(cursor.erase(""), end="")

    def _setup_cursor_for_normal_printing(self, image_height: int) -> None:
        """Move the cursor below the UI, clear any colors, and erase anything below the UI.

        This allows usage of print outside the class in a well-defined manner, appearing below the UI and being reset
        every after every call to this function.
        """
        print(self._cursor_goto(Vec(image_height, 0)) + color.fx.reset + cursorx.erase_to_end(""))

    def _draw_array_row(self, top_left: Vec, array_row: NDArray[np.uint8]) -> None:
        print(
            self._cursor_goto(top_left)
            + "".join(
                self._get_color_str(
                    int(color_index),
                    (top_left.y, top_left.x + x_index),
                )
                + " " * CLI._PIXEL_WIDTH
                for x_index, color_index in enumerate(array_row)
            ),
            end="",
        )

    def _draw_pixel(self, position: Vec, color_index: int, text: str) -> None:
        print(
            self._cursor_goto(position)
            + self._get_color_str(int(color_index), astuple(position))
            + ((text and (self._text_fx + text.ljust(CLI._PIXEL_WIDTH))) or " " * CLI._PIXEL_WIDTH),
            end="",
        )

    def _get_color_str(self, color_index: int, position: tuple[int, int]) -> str:
        assert self._dynamic_layer is not None
        return self._color_palette.get_color(index=color_index, dynamic_layer_value=int(self._dynamic_layer[position]))
