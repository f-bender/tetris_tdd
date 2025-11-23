import colorsys
import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from itertools import count
from typing import Literal, NamedTuple, Self, cast

import numpy as np

from tetris.ansi_extensions import color as colorx

_LOGGER = logging.getLogger(__name__)

_DYNAMIC_BACKGROUND_COLORMAP_DEFAULT_LENGTH = 500


class _Colors(NamedTuple):
    # we allow 10 different shades to represent blocks that are placed but not yet (four-)colored
    outer_bg_progress_1: str
    outer_bg_progress_2: str
    outer_bg_progress_3: str
    outer_bg_progress_4: str
    outer_bg_progress_5: str
    outer_bg_progress_6: str
    outer_bg_progress_7: str
    outer_bg_progress_8: str
    outer_bg_progress_9: str
    outer_bg_progress_10: str
    # the 4 colors shown as the result of the four-colorizer
    outer_bg_1: str
    outer_bg_2: str
    outer_bg_3: str
    outer_bg_4: str
    # the 2 colors of the checkerboard-patterned board background
    board_bg_1: str
    board_bg_2: str
    # the 2 colors of the checkerboard-patterned board background, when showing the ghost piece
    board_bg_1_ghost: str
    board_bg_2_ghost: str
    # there are 7 different block types, each with a different color
    block_1: str
    block_2: str
    block_3: str
    block_4: str
    block_5: str
    block_6: str
    block_7: str
    # neutral color for blocks of "unknown origin", not from the standard set (e.g. placed by tetris99 rule)
    block_neutral: str
    # background of score and next block display
    display_bg: str
    # animation colors
    tetris_sparkle: str
    # background that has not (yet) been filled or (four-)colored
    empty: str


class BackgroundColorType(Enum):
    NORMAL = auto()
    SHINY = auto()
    DYNAMIC = auto()

    @classmethod
    def random(cls, shiny_probability: float = 0.05, dynamic_probability: float = 0.05) -> "BackgroundColorType":
        if shiny_probability < 0 or dynamic_probability < 0 or shiny_probability + dynamic_probability > 1:
            msg = f"Invalid probability values: {shiny_probability = }, {dynamic_probability = }"
            raise ValueError(msg)

        rand = random.random()
        if rand < shiny_probability:
            return cls.SHINY

        if rand < shiny_probability + dynamic_probability:
            return cls.DYNAMIC

        return cls.NORMAL


@dataclass(slots=True)
class ColorPalette:
    colors: _Colors

    # Sequences of ANSI codes for colors to be cycled through. The provided dynamic_layer_value in get_color will be
    # used to index into these sequences to get the color (modulo'd with the sequences length). The first and last color
    # should be considered adjacent (i.e. be similar to avoid a sudden jump in color).
    # Note: the length of the colormap effectively determines the speed of the animation (longer = slower).
    dynamic_colormap_background: tuple[str, ...]
    dynamic_colormap_powerup: tuple[str, ...]

    # save the function used to generate the ANSI colors from RGB values, so we can use it again when randomizing
    color_fn: Callable[[int, int, int], str]
    # how far to offset the 4 different background indexes from one another in their colormap cycle
    dynamic_background_relative_colormap_offset_step: float = 1 / 8

    _index_counter = count(len(_Colors._fields))

    DYNAMIC_BACKGROUND_INDEX_0 = next(_index_counter)
    DYNAMIC_BACKGROUND_INDEX_1 = next(_index_counter)
    DYNAMIC_BACKGROUND_INDEX_2 = next(_index_counter)
    DYNAMIC_BACKGROUND_INDEX_3 = next(_index_counter)

    STATIC_DYNAMIC_IDX_OFFSET = DYNAMIC_BACKGROUND_INDEX_0 - _Colors._fields.index("outer_bg_1")

    DYNAMIC_POWERUP_INDEX = next(_index_counter)

    # we use np.uint8's for indexing, so the last used value must still be within the range
    assert next(_index_counter) - 1 <= np.iinfo(np.uint8).max
    del _index_counter

    def get_color(self, index: int, dynamic_layer_value: int | None = None) -> str:
        if index < self.DYNAMIC_BACKGROUND_INDEX_0:
            # the index references a static color
            return self.colors[index]

        assert dynamic_layer_value is not None, (
            f"{index} is a dynamic color index, but no dynamic layer value was provided"
        )

        if index <= self.DYNAMIC_BACKGROUND_INDEX_3:
            return self.dynamic_colormap_background[
                (
                    dynamic_layer_value
                    + round(
                        (index - self.DYNAMIC_BACKGROUND_INDEX_0)
                        * self.dynamic_background_relative_colormap_offset_step
                        * len(self.dynamic_colormap_background)
                    )
                )
                % len(self.dynamic_colormap_background)
            ]

        assert index == self.DYNAMIC_POWERUP_INDEX, f"{index} is out of range"
        return self.dynamic_colormap_powerup[dynamic_layer_value % len(self.dynamic_colormap_powerup)]

    @classmethod
    def default(cls) -> Self:
        return cls.from_rgb(
            # progressive shades of gray for outer background during block placement
            **{f"outer_bg_progress_{i}": (127 + 10 * (i - 5),) * 3 for i in range(1, 11)},  # type: ignore[arg-type]
            # "christmats themed" outer background colors, bright and dark greens and reds
            outer_bg_1=(46, 0, 2),
            outer_bg_2=(39, 85, 10),
            outer_bg_3=(123, 1, 6),
            outer_bg_4=(15, 33, 4),
            # dark gray board background
            board_bg_1=(50, 50, 50),
            board_bg_2=(30, 30, 30),
            board_bg_1_ghost=(120, 120, 120),
            board_bg_2_ghost=(100, 100, 100),
            # standard Tetris block colors
            block_1=(160, 1, 241),  # T
            block_2=(248, 230, 8),  # O
            block_3=(0, 255, 255),  # I
            block_4=(239, 130, 1),  # L
            block_5=(2, 241, 2),  # S
            block_6=(51, 153, 255),  # J
            block_7=(240, 0, 1),  # Z
            # light gray neutral block color
            block_neutral=(200, 200, 200),
            # yellow tetris sparkle animation
            tetris_sparkle=(255, 255, 0),
            # dark gray background for the text, black for emptiness
            display_bg=(50, 50, 50),
            empty=(0, 0, 0),
            dynamic_colormap_background=cls.random_colormap(),
            dynamic_colormap_powerup=cls.rainbow_colormap(length=200),
        )

    @classmethod
    def from_rgb(  # noqa: PLR0913
        cls,
        *,
        outer_bg_progress_1: tuple[int, int, int],
        outer_bg_progress_2: tuple[int, int, int],
        outer_bg_progress_3: tuple[int, int, int],
        outer_bg_progress_4: tuple[int, int, int],
        outer_bg_progress_5: tuple[int, int, int],
        outer_bg_progress_6: tuple[int, int, int],
        outer_bg_progress_7: tuple[int, int, int],
        outer_bg_progress_8: tuple[int, int, int],
        outer_bg_progress_9: tuple[int, int, int],
        outer_bg_progress_10: tuple[int, int, int],
        outer_bg_1: tuple[int, int, int],
        outer_bg_2: tuple[int, int, int],
        outer_bg_3: tuple[int, int, int],
        outer_bg_4: tuple[int, int, int],
        board_bg_1: tuple[int, int, int],
        board_bg_2: tuple[int, int, int],
        board_bg_1_ghost: tuple[int, int, int],
        board_bg_2_ghost: tuple[int, int, int],
        block_1: tuple[int, int, int],
        block_2: tuple[int, int, int],
        block_3: tuple[int, int, int],
        block_4: tuple[int, int, int],
        block_5: tuple[int, int, int],
        block_6: tuple[int, int, int],
        block_7: tuple[int, int, int],
        block_neutral: tuple[int, int, int],
        display_bg: tuple[int, int, int],
        tetris_sparkle: tuple[int, int, int],
        empty: tuple[int, int, int],
        dynamic_colormap_background: Iterable[tuple[int, int, int]],
        dynamic_colormap_powerup: Iterable[tuple[int, int, int]],
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
    ) -> Self:
        return cls(
            colors=_Colors(
                outer_bg_progress_1=color_fn(*outer_bg_progress_1),
                outer_bg_progress_2=color_fn(*outer_bg_progress_2),
                outer_bg_progress_3=color_fn(*outer_bg_progress_3),
                outer_bg_progress_4=color_fn(*outer_bg_progress_4),
                outer_bg_progress_5=color_fn(*outer_bg_progress_5),
                outer_bg_progress_6=color_fn(*outer_bg_progress_6),
                outer_bg_progress_7=color_fn(*outer_bg_progress_7),
                outer_bg_progress_8=color_fn(*outer_bg_progress_8),
                outer_bg_progress_9=color_fn(*outer_bg_progress_9),
                outer_bg_progress_10=color_fn(*outer_bg_progress_10),
                outer_bg_1=color_fn(*outer_bg_1),
                outer_bg_2=color_fn(*outer_bg_2),
                outer_bg_3=color_fn(*outer_bg_3),
                outer_bg_4=color_fn(*outer_bg_4),
                board_bg_1=color_fn(*board_bg_1),
                board_bg_2=color_fn(*board_bg_2),
                board_bg_1_ghost=color_fn(*board_bg_1_ghost),
                board_bg_2_ghost=color_fn(*board_bg_2_ghost),
                block_1=color_fn(*block_1),
                block_2=color_fn(*block_2),
                block_3=color_fn(*block_3),
                block_4=color_fn(*block_4),
                block_5=color_fn(*block_5),
                block_6=color_fn(*block_6),
                block_7=color_fn(*block_7),
                block_neutral=color_fn(*block_neutral),
                display_bg=color_fn(*display_bg),
                tetris_sparkle=color_fn(*tetris_sparkle),
                empty=color_fn(*empty),
            ),
            dynamic_colormap_background=tuple(color_fn(r, g, b) for r, g, b in dynamic_colormap_background),
            dynamic_colormap_powerup=tuple(color_fn(r, g, b) for r, g, b in dynamic_colormap_powerup),
            color_fn=color_fn,
        )

    @classmethod
    @cache
    def index_of_color(
        cls,
        color_name: Literal[
            "outer_bg_progress_1",
            "outer_bg_progress_2",
            "outer_bg_progress_3",
            "outer_bg_progress_4",
            "outer_bg_progress_5",
            "outer_bg_progress_6",
            "outer_bg_progress_7",
            "outer_bg_progress_8",
            "outer_bg_progress_9",
            "outer_bg_progress_10",
            "outer_bg_1",
            "outer_bg_2",
            "outer_bg_3",
            "outer_bg_4",
            "board_bg_1",
            "board_bg_2",
            "board_bg_1_ghost",
            "board_bg_2_ghost",
            "block_1",
            "block_2",
            "block_3",
            "block_4",
            "block_5",
            "block_6",
            "block_7",
            "block_neutral",
            "display_bg",
            "tetris_sparkle",
            "empty",
        ],
    ) -> int:
        try:
            return _Colors._fields.index(color_name)
        except ValueError as e:
            msg = f"Invalid color name: '{color_name}'"
            raise ValueError(msg) from e

    @staticmethod
    @cache
    def block_color_index_offset() -> int:
        return ColorPalette.index_of_color("block_1")

    @staticmethod
    @cache
    def board_bg_index_offset() -> int:
        return ColorPalette.index_of_color("board_bg_1")

    @staticmethod
    @cache
    def board_bg_ghost_index_offset() -> int:
        return ColorPalette.index_of_color("board_bg_1_ghost")

    @staticmethod
    @cache
    def outer_bg_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_1")

    @staticmethod
    @cache
    def outer_bg_progress_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_progress_1")

    def randomize_outer_bg_colors(
        self,
        *,
        bg_color_type: BackgroundColorType = BackgroundColorType.NORMAL,
        dynamic_background_speed_factor: float | None = 1,
    ) -> None:
        if bg_color_type is BackgroundColorType.DYNAMIC:
            if dynamic_background_speed_factor is None:
                msg = "a dynamic_background_speed_factor is required when bg_color_type is DYNAMIC"
                raise ValueError(msg)

            self.dynamic_colormap_background = tuple(
                self.color_fn(r, g, b)
                for r, g, b in self.random_colormap(
                    length=round(_DYNAMIC_BACKGROUND_COLORMAP_DEFAULT_LENGTH / dynamic_background_speed_factor)
                )
            )
            return

        rgb_1, rgb_2, rgb_3, rgb_4 = self._generate_random_outer_bg_colors(
            shiny=bg_color_type is BackgroundColorType.SHINY
        )
        self.colors = self.colors._replace(
            outer_bg_1=self.color_fn(*rgb_1),
            outer_bg_2=self.color_fn(*rgb_2),
            outer_bg_3=self.color_fn(*rgb_3),
            outer_bg_4=self.color_fn(*rgb_4),
        )

    @staticmethod
    def _generate_random_outer_bg_colors(
        *, shiny: bool = False
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        # 1 in 1000 chance for a "shiny" palette (fully random colors)
        if shiny:
            shiny_rgb_1, shiny_rgb_2, shiny_rgb_3, shiny_rgb_4 = (
                tuple(random.randint(0, 255) for _ in range(3)) for _ in range(4)
            )
            _LOGGER.debug(
                "Generated shiny outer background palette RGB values: (%s, %s, %s, %s)",
                shiny_rgb_1,
                shiny_rgb_2,
                shiny_rgb_3,
                shiny_rgb_4,
            )
            return shiny_rgb_1, shiny_rgb_2, shiny_rgb_3, shiny_rgb_4  # type: ignore[return-value]

        hue_1 = random.random()
        hue_2 = random.uniform(hue_1 + 0.25, hue_1 + 0.75) % 1.0

        saturation_1 = random.uniform(0.8, 1.0)
        saturation_2 = random.uniform(0.8, 1.0)

        value_1 = random.uniform(0.4, 0.5)
        value_2 = random.uniform(0.2, min(value_1 / 2, 0.3))

        _LOGGER.debug(
            "Generated outer background palette HSV values: "
            "hue_1=%.2f, saturation_1=%.2f, value_1=%.2f; "
            "hue_2=%.2f, saturation_2=%.2f, value_2=%.2f",
            hue_1,
            saturation_1,
            value_1,
            hue_2,
            saturation_2,
            value_2,
        )

        rgb_1 = colorsys.hsv_to_rgb(hue_1, saturation_1, value_1)
        rgb_2 = colorsys.hsv_to_rgb(hue_1, saturation_1, value_2)
        rgb_3 = colorsys.hsv_to_rgb(hue_2, saturation_2, value_1)
        rgb_4 = colorsys.hsv_to_rgb(hue_2, saturation_2, value_2)

        return tuple(tuple(round(c * 255) for c in rgb) for rgb in (rgb_1, rgb_2, rgb_3, rgb_4))  # type: ignore[return-value]

    @staticmethod
    def rainbow_colormap(
        length: int = _DYNAMIC_BACKGROUND_COLORMAP_DEFAULT_LENGTH,
        *,
        saturation: float = 1.0,
        value: float = 0.7,
        reverse: bool = False,
    ) -> Iterable[tuple[int, int, int]]:
        cycle_range = range(length)

        return (
            cast(
                "tuple[int, int, int]",
                tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h=cycle_value / length, s=saturation, v=value)),
            )
            for cycle_value in (reversed(cycle_range) if reverse else cycle_range)
        )

    @staticmethod
    def colorcet_colormap(
        length: int = _DYNAMIC_BACKGROUND_COLORMAP_DEFAULT_LENGTH, *, name: str | None = None
    ) -> Iterable[tuple[int, int, int]] | None:
        try:
            from tetris.ui.cli.colorcet_colormaps import get_colorcet_colormap  # noqa: PLC0415
        except ImportError:
            _LOGGER.info("Can't create colorcet colormap due to missing dependency")
            return None

        cmap_float_array = get_colorcet_colormap(length=length, name=name)
        # limit the brightness to ensure the tetris board still stands out
        cmap_float_array *= 0.8
        return (tuple(round(c * 255) for c in rgb) for rgb in cmap_float_array)

    @classmethod
    def random_colormap(
        cls, length: int = _DYNAMIC_BACKGROUND_COLORMAP_DEFAULT_LENGTH, *, p_colorcet: float = 0.9
    ) -> Iterable[tuple[int, int, int]]:
        if random.random() < p_colorcet and (colorcet_colormap := cls.colorcet_colormap(length=length)):
            return colorcet_colormap

        saturation = random.uniform(0.9, 1.0)
        value = random.uniform(0.7, 0.8)
        reverse = random.choice([True, False])
        return cls.rainbow_colormap(length=length, saturation=saturation, value=value, reverse=reverse)
