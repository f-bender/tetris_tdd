import colorsys
import logging
import random
from functools import cache
from typing import Literal, NamedTuple, Self

from tetris.ansi_extensions import color as colorx

_LOGGER = logging.getLogger(__name__)


class ColorPalette(NamedTuple):
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
    board_bg: str
    board_bg_alt: str
    # there are 7 different block types, each with a different color
    block_1: str
    block_2: str
    block_3: str
    block_4: str
    block_5: str
    block_6: str
    block_7: str
    block_neutral: str
    # background of score and next block display
    display_bg: str
    # animation colors
    tetris_sparkle: str
    # background that has not (yet) been filled or (four-)colored
    empty: str = colorx.bg.rgb_palette(0, 0, 0)
    # save the mode being used
    mode: Literal["palette", "truecolor"] = "truecolor"

    @classmethod
    def from_rgb(  # noqa: PLR0913
        cls,
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
        board_bg: tuple[int, int, int],
        board_bg_alt: tuple[int, int, int],
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
        empty: tuple[int, int, int] = (0, 0, 0),
        *,
        mode: Literal["palette", "truecolor"] = "truecolor",
    ) -> Self:
        color_fn = colorx.bg.rgb_palette if mode == "palette" else colorx.bg.rgb_truecolor
        return cls(
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
            board_bg=color_fn(*board_bg),
            board_bg_alt=color_fn(*board_bg_alt),
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
        )

    @classmethod
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
            "board_bg",
            "board_bg_alt",
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
            return cls._fields.index(color_name)
        except ValueError as e:
            msg = f"Invalid color name: '{color_name}'"
            raise ValueError(msg) from e

    @cache
    @staticmethod
    def block_color_index_offset() -> int:
        return ColorPalette.index_of_color("block_1")

    @cache
    @staticmethod
    def outer_bg_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_1")

    @cache
    @staticmethod
    def outer_bg_progress_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_progress_1")

    def with_randomized_outer_bg_palette(self) -> Self:
        color_fn = colorx.bg.rgb_palette if self.mode == "palette" else colorx.bg.rgb_truecolor
        rgb_1, rgb_2, rgb_3, rgb_4 = self._generate_random_outer_bg_palette()
        return self._replace(
            outer_bg_1=color_fn(*rgb_1),
            outer_bg_2=color_fn(*rgb_2),
            outer_bg_3=color_fn(*rgb_3),
            outer_bg_4=color_fn(*rgb_4),
        )

    @staticmethod
    def _generate_random_outer_bg_palette() -> tuple[
        tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]
    ]:
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
