import colorsys
import contextlib
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any, Literal, NamedTuple, Self

import numpy as np

from tetris.ansi_extensions import color as colorx

_LOGGER = logging.getLogger(__name__)


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


@dataclass
class ColorPalette:
    colors: _Colors

    # save the function used to generate the ANSI colors from RGB values, so we can use it again when randomizing
    color_fn: Callable[[int, int, int], str]

    # saturation and value of the rainbow colors used for the rainbow animation
    rainbow_saturation: float = 1.0
    rainbow_value: float = 0.9

    DYNAMIC_BACKGROUND_INDEX_0 = len(_Colors._fields)
    DYNAMIC_BACKGROUND_INDEX_1 = len(_Colors._fields) + 1
    DYNAMIC_BACKGROUND_INDEX_2 = len(_Colors._fields) + 2
    DYNAMIC_BACKGROUND_INDEX_3 = len(_Colors._fields) + 3

    DYNAMIC_POWERUP_INDEX_3 = len(_Colors._fields) + 4

    # we use np.uint8's for indexing, so this must still be within the range
    assert DYNAMIC_POWERUP_INDEX_3 <= np.iinfo(np.uint8).max  # noqa: SIM300

    @cached_property
    def rainbow_colors(self) -> tuple[str, ...]:
        return tuple(
            colorx.bg.rgb_truecolor(
                *(
                    round(c * 255)
                    for c in colorsys.hsv_to_rgb(h=cycle_value / 256, s=self.rainbow_saturation, v=self.rainbow_value)
                )
            )
            for cycle_value in range(256)
        )

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name in ("rainbow_saturation", "rainbow_value"):
            # invalidate cached rainbow colors
            with contextlib.suppress(AttributeError):
                del self.rainbow_colors

        object.__setattr__(self, name, value)

    def __getitem__(self, i: int) -> str:
        return self.colors[i]

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
            # dark gray background for the text
            display_bg=(50, 50, 50),
        )

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
        empty: tuple[int, int, int] = (0, 0, 0),
        *,
        color_fn: Callable[[int, int, int], str] = colorx.bg.rgb_truecolor,
    ) -> Self:
        return cls(
            _Colors(
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
            color_fn=color_fn,
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

    @cache
    @staticmethod
    def block_color_index_offset() -> int:
        return ColorPalette.index_of_color("block_1")

    @cache
    @staticmethod
    def board_bg_index_offset() -> int:
        return ColorPalette.index_of_color("board_bg_1")

    @cache
    @staticmethod
    def board_bg_ghost_index_offset() -> int:
        return ColorPalette.index_of_color("board_bg_1_ghost")

    @cache
    @staticmethod
    def outer_bg_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_1")

    @cache
    @staticmethod
    def outer_bg_progress_index_offset() -> int:
        return ColorPalette.index_of_color("outer_bg_progress_1")

    def randomize_outer_bg_colors(self, *, shiny: bool = False) -> None:
        rgb_1, rgb_2, rgb_3, rgb_4 = self._generate_random_outer_bg_colors(shiny=shiny)
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
