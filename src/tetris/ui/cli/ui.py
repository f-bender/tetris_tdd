import atexit
import colorsys
import inspect
import logging
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import astuple
from enum import Enum, auto
from functools import cached_property, lru_cache
from math import ceil
from typing import TYPE_CHECKING, Literal, Self, cast, get_args, get_origin

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import color as colorx
from tetris.ansi_extensions import cursor as cursorx
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.space_filling_coloring import fill_and_colorize
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


class BackgroundColorType(Enum):
    NORMAL = auto()
    SHINY = auto()
    DYNAMIC = auto()

    @classmethod
    def random(cls, shiny_probability: float = 0.001, dynamic_probability: float = 0.001) -> "BackgroundColorType":
        if shiny_probability < 0 or dynamic_probability < 0 or shiny_probability + dynamic_probability > 1:
            msg = f"Invalid probability values: {shiny_probability = }, {dynamic_probability = }"
            raise ValueError(msg)

        rand = random.random()
        if rand < shiny_probability:
            return cls.SHINY

        if rand < shiny_probability + dynamic_probability:
            return cls.DYNAMIC

        return cls.NORMAL


class DynamicLayer:
    def __init__(self, initial_layer: NDArray[np.uint32], colormap: Sequence[str]) -> None:
        """Initialize the DynamicLayer.

        Args:
            initial_layer: Numpy array with the initial state of the dynamic layer which fully describes the "shape" of
                the dynamic movement.
            colormap: Sequence of ANSI codes for colors to be cycled through. The values in the layer will be used to
                index into this sequence to get the color at that position (modulo'd with the sequences length). The
                first and last color should be considered adjacent (i.e. be similar to avoid a sudden jump in color).
                Note: the length of the colormap effectively determines the speed of the animation (longer = slower).
        """
        self.layer = initial_layer
        self.colormap = colormap

    def update(self) -> None:
        # NOTE: this *could* technically overflow, but it would have to run for 2 years straight (at 60 fps),
        # and even then it wouldn't break but just look weird for a few seconds
        self.layer += 1

    def get_color(self, position: tuple[int, int], relative_offset: float) -> str:
        return self.colormap[
            (int(self.layer[position]) + round(relative_offset * len(self.colormap))) % len(self.colormap)
        ]

    @staticmethod
    def mix_layers(
        lhs: NDArray[np.uint32], rhs: NDArray[np.uint32], /, *, method: Literal["min", "max"] = "min"
    ) -> NDArray[np.uint32]:
        if lhs.shape != rhs.shape:
            msg = "Can't mix DynamicLayers of differing shapes"
            raise ValueError(msg)

        return (np.minimum if method == "min" else np.maximum)(lhs, rhs)

    # layer constructors

    @staticmethod
    def cardinal_layer(
        size: tuple[int, int], direction: Literal["up", "down", "left", "right"] = "right"
    ) -> NDArray[np.uint32]:
        match direction:
            case "left":
                return np.tile(np.arange(size[1], dtype=np.uint32), (size[0], 1))
            case "right":
                return np.tile(np.arange(size[1], dtype=np.uint32)[::-1], (size[0], 1))
            case "up":
                return np.tile(np.arange(size[0], dtype=np.uint32)[:, np.newaxis], size[1])
            case "down":
                return np.tile(np.arange(size[0], dtype=np.uint32)[::-1, np.newaxis], size[1])

    @staticmethod
    def diagonal_layer(
        size: tuple[int, int],
        y_direction: Literal["up", "down"] = "down",
        x_direction: Literal["left", "right"] = "right",
    ) -> NDArray[np.uint32]:
        y_range = np.arange(size[0], dtype=np.uint32)
        if y_direction == "down":
            y_range = y_range[::-1]

        x_range = np.arange(size[1], dtype=np.uint32)
        if x_direction == "right":
            x_range = x_range[::-1]

        return np.add.outer(y_range, x_range)

    @staticmethod
    def circular_layer(
        size: tuple[int, int],
        direction: Literal["inward", "outward"] = "outward",
        y_center: Literal["top", "center", "bottom", "random"] = "random",
        x_center: Literal["left", "center", "right", "random"] = "random",
    ) -> NDArray[np.uint32]:
        match y_center:
            case "top":
                y_center_idx = 0
            case "center":
                y_center_idx = size[0] // 2
            case "bottom":
                y_center_idx = size[0]
            case "random":
                y_center_idx = random.randrange(size[0])

        match x_center:
            case "left":
                x_center_idx = 0
            case "center":
                x_center_idx = size[1] // 2
            case "right":
                x_center_idx = size[1]
            case "random":
                x_center_idx = random.randrange(size[1])

        x_coordinate_array = np.tile(np.arange(size[1]), (size[0], 1))
        y_coordinate_array = np.tile(np.arange(size[0])[:, np.newaxis], size[1])

        distance_array = np.sqrt(
            (x_coordinate_array - x_center_idx) ** 2 + (y_coordinate_array - y_center_idx) ** 2
        ).astype(np.uint32)
        if direction == "outward":
            distance_array = np.max(distance_array) - distance_array

        return distance_array

    @classmethod
    def random_layer(cls, size: tuple[int, int], *, n_mixed_layers: int = 1) -> NDArray[np.uint32]:
        if n_mixed_layers < 1:
            msg = "n_mixed_layers must at least be 1"
            raise ValueError(msg)

        dynamic_layer: NDArray[np.uint32] | None = None
        for _ in range(n_mixed_layers):
            constructor = random.choice([cls.cardinal_layer, cls.diagonal_layer, cls.circular_layer])

            parameters = inspect.signature(constructor).parameters

            kwargs = {}
            for name, param in parameters.items():
                if name == "size":
                    continue

                assert get_origin(param.annotation) is Literal
                kwargs[name] = random.choice(get_args(param.annotation))

            new_dynamic_layer = constructor(size=size, **kwargs)

            if dynamic_layer is None:
                dynamic_layer = new_dynamic_layer
            else:
                dynamic_layer = cls.mix_layers(dynamic_layer, new_dynamic_layer, method=random.choice(["min", "max"]))

        assert dynamic_layer is not None
        return dynamic_layer

    # colormap constructors

    @staticmethod
    def rainbow_colormap(
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
        length: int = 500,
        *,
        saturation: float = 1.0,
        value: float = 0.7,
    ) -> tuple[str, ...]:
        return tuple(
            color_fn(*(round(c * 255) for c in colorsys.hsv_to_rgb(h=cycle_value / length, s=saturation, v=value)))
            for cycle_value in range(length)
        )

    @staticmethod
    def colorcet_colormap(
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
        length: int = 500,
        *,
        name: str | None = None,
    ) -> tuple[str, ...] | None:
        try:
            from tetris.ui.cli.colorcet_colormaps import get_colorcet_colormap  # noqa: PLC0415
        except ImportError:
            LOGGER.info("Can't create colorcet colormap due to missing dependency")
            return None

        cmap_float_array = get_colorcet_colormap(length=length, name=name)
        # limit the brightness to ensure the tetris board still stand out
        cmap_float_array *= 0.8
        return tuple(color_fn(*(round(c * 255) for c in rgb)) for rgb in cmap_float_array)

    @classmethod
    def random_colormap(
        cls,
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
        length: int = 500,
        *,
        p_colorcet: float = 0.9,
    ) -> tuple[str, ...]:
        if random.random() < p_colorcet and (
            colorcet_colormap := cls.colorcet_colormap(color_fn=color_fn, length=length)
        ):
            return colorcet_colormap

        saturation = random.uniform(0.9, 1.0)
        value = random.uniform(0.7, 0.8)
        return cls.rainbow_colormap(color_fn=color_fn, length=length, saturation=saturation, value=value)

    # random construction of whole DynamicLayer object

    @classmethod
    def random(
        cls,
        size: tuple[int, int],
        *,
        n_mixed_layers: int = 1,
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
        colormap_length: int = 500,
        p_colorcet: float = 0.9,
    ) -> Self:
        return cls(
            initial_layer=cls.random_layer(size=size, n_mixed_layers=n_mixed_layers),
            colormap=cls.random_colormap(color_fn=color_fn, length=colormap_length, p_colorcet=p_colorcet),
        )


class CLI(UI):
    _PIXEL_WIDTH = 2  # how many terminal characters together form one pixel
    _FRAME_WIDTH = 8  # width of the static frame around and between the single games' UIs, in pixels

    _EMOJI_THRESHOLD = 0x2500  # characters with unicodes higher than this are considered emojis

    # if there are more pixel changes than this in one row, the whole row is re-drawn (not just the changed pixels)
    # (this is roughly the threshold where whole-row draws become more efficient)
    _MAX_PIXELS_PER_ROW_TO_DELTA_DRAW = 10

    _DYNAMIC_BACKGROUND_RELATIVE_COLORMAP_OFFSET = 1 / 8

    def __init__(
        self,
        *,
        color_palette: ColorPalette | None = None,
        target_aspect_ratio: float = 16 / 9,
        randomize_background_colors_on_levelup: bool = False,
    ) -> None:
        self._last_image_buffer: NDArray[np.uint8] | None = None
        self._last_text_buffer: NDArray[np.str_] | None = None
        self._board_background: NDArray[np.uint8] | None = None
        self._outer_background: NDArray[np.uint8] | None = None

        self._single_game_ui: SingleGameUI | None = None
        self._game_ui_offsets: list[Vec] | None = None

        self._color_palette = color_palette or ColorPalette.default()

        # TODO change probabilities back
        bg_color_type = BackgroundColorType.random(dynamic_probability=1, shiny_probability=0)
        self._dynamic_background_during_startup = bg_color_type is BackgroundColorType.DYNAMIC
        if not self._dynamic_background_during_startup:
            self._color_palette.randomize_outer_bg_colors(shiny=bg_color_type is BackgroundColorType.SHINY)

        self._buffered_print = BufferedPrint()
        self._startup_animation_iter: (
            Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]
            | None
        ) = None
        self._startup_finished: bool = False

        self._target_aspect_ratio = target_aspect_ratio

        self._last_terminal_size = os.get_terminal_size()

        self._randomize_background_colors_on_levelup = randomize_background_colors_on_levelup
        self._level: int | None = None
        self._dynamic_layer: DynamicLayer | None = None

    @cached_property
    def total_size(self) -> tuple[int, int]:
        if self._game_ui_offsets is None or self._single_game_ui is None:
            msg = "board UI not initialized, cannot get total_size before initialize() is called!"
            raise RuntimeError(msg)

        board_ui_height, board_ui_width = self._single_game_ui.total_size
        total_height = max(offset.y for offset in self._game_ui_offsets) + board_ui_height + self._FRAME_WIDTH
        total_width = max(offset.x for offset in self._game_ui_offsets) + board_ui_width + self._FRAME_WIDTH

        return total_height, total_width

    @staticmethod
    def _cursor_goto(vec: Vec) -> str:
        # + 1 to make the interface 0-based (index of top CLI row, and left CLI column is actually 1, not 0)
        return cursor.goto(vec.y + 1, vec.x * CLI._PIXEL_WIDTH + 1)

    def initialize(self, board_height: int, board_width: int, num_boards: int) -> None:
        if num_boards <= 0:
            msg = "num_boards must be greater than 0"
            raise ValueError(msg)

        self._single_game_ui = SingleGameUI(board_height, board_width)
        self._initialize_board_ui_offsets(num_boards)
        self._initialize_terminal()

        self._dynamic_layer = DynamicLayer.random(size=self.total_size, n_mixed_layers=3, colormap_length=1_000)

    def _initialize_board_ui_offsets(self, num_games: int) -> None:
        assert self._single_game_ui is not None

        game_ui_height, game_ui_width = self._single_game_ui.total_size

        num_rows, num_cols = self._compute_num_rows_cols(num_games)

        self._game_ui_offsets = [
            Vec(
                self._FRAME_WIDTH + y * (game_ui_height + self._FRAME_WIDTH),
                self._FRAME_WIDTH + x * (game_ui_width + self._FRAME_WIDTH),
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
        height_added_per_game = single_game_ui_height + self._FRAME_WIDTH
        width_added_per_game = single_game_ui_width + self._FRAME_WIDTH

        # solution from WolframAlpha, prompted with
        # `(c * w + f) / (r * h + f) = a, r * c = n, solve for r, c`
        # where c: num_cols, w: width_added_per_game, f: frame_width, r: num_rows, h: height_added_per_game,
        #       a: target_aspect_ratio, n: num_games
        # (`ceil` added myself to make it integers)
        num_cols = ceil(
            (
                (
                    (self._target_aspect_ratio - 1) ** 2 * self._FRAME_WIDTH**2
                    + 4 * self._target_aspect_ratio * height_added_per_game * num_games * width_added_per_game
                )
                ** 0.5
                + (self._target_aspect_ratio - 1) * self._FRAME_WIDTH
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
        # return True
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
            self._outer_background = (
                self._create_outer_background_mask().astype(np.uint8)
                * ColorPalette.DYNAMIC_BACKGROUND_INDEX_0  # TODO: remove
            )

        self._dynamic_layer.update()

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

        static_dynamic_idx_offset = ColorPalette.DYNAMIC_BACKGROUND_INDEX_0 - ColorPalette.outer_bg_index_offset()

        bg_color_type = BackgroundColorType.random()

        # reset from dynamic background, in case we had dynamic background before
        if np.any(self._outer_background >= ColorPalette.DYNAMIC_BACKGROUND_INDEX_0):
            self._outer_background = np.where(
                self._outer_background != ColorPalette.index_of_color("empty"),
                self._outer_background - static_dynamic_idx_offset,
                self._outer_background,
            )
            # and make sure we don't use dynamic twice in a row (fall back to normal)
            if bg_color_type is BackgroundColorType.DYNAMIC:
                bg_color_type = BackgroundColorType.NORMAL

        match bg_color_type:
            case BackgroundColorType.DYNAMIC:
                self._outer_background = np.where(
                    self._outer_background != ColorPalette.index_of_color("empty"),
                    self._outer_background + static_dynamic_idx_offset,
                    self._outer_background,
                )
                # add a subtle random variation in saturation and value
                self._color_palette.rainbow_saturation = random.uniform(0.9, 1.0)
                self._color_palette.rainbow_value = random.uniform(0.8, 0.9)
            case BackgroundColorType.SHINY | BackgroundColorType.NORMAL:
                self._color_palette.randomize_outer_bg_colors(shiny=bg_color_type is BackgroundColorType.SHINY)

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

    def _draw_array(self, top_left: Vec, array: NDArray[np.uint8]) -> None:
        # NOTE: we don't draw text while drawing entire rows at a time since emojis in text can mess up the spacing
        for idx, row in enumerate(array):
            self._draw_array_row(top_left=top_left + Vec(idx, 0), array_row=row)

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
            + ((text and text.ljust(CLI._PIXEL_WIDTH)) or " " * CLI._PIXEL_WIDTH),
            end="",
        )

    def _get_color_str(self, color_index: int, position: tuple[int, int]) -> str:
        if color_index < ColorPalette.DYNAMIC_BACKGROUND_INDEX_0:
            return self._color_palette[color_index]

        assert self._dynamic_layer is not None

        # TODO make use of dynamic_layer.get_color; think about where to build the offset stuff into (or whether to
        # just keep it here)
        return self._dynamic_layer.get_color(
            position=position,
            relative_offset=(color_index - ColorPalette.DYNAMIC_BACKGROUND_INDEX_0)
            * self._DYNAMIC_BACKGROUND_RELATIVE_COLORMAP_OFFSET,
        )
