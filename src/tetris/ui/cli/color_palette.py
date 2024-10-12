from typing import Literal, NamedTuple, Self

from tetris.ansi_extensions import color as colorx


class ColorPalette(NamedTuple):
    outer_bg_progress: str
    outer_bg_1: str  # needs to be at index 1
    outer_bg_2: str  # needs to be at index 2
    outer_bg_3: str  # needs to be at index 3
    outer_bg_4: str  # needs to be at index 4
    board_bg: str
    board_bg_alt: str
    board_fg: str
    empty: str = colorx.bg.rgb_palette(0, 0, 0)

    @classmethod
    def from_rgb(  # noqa: PLR0913
        cls,
        outer_bg_progress: tuple[int, int, int],
        outer_bg_1: tuple[int, int, int],
        outer_bg_2: tuple[int, int, int],
        outer_bg_3: tuple[int, int, int],
        outer_bg_4: tuple[int, int, int],
        board_bg: tuple[int, int, int],
        board_bg_alt: tuple[int, int, int],
        board_fg: tuple[int, int, int],
        empty: tuple[int, int, int] = (0, 0, 0),
        *,
        mode: Literal["palette", "truecolor"] = "truecolor",
    ) -> Self:
        color_fn = colorx.bg.rgb_palette if mode == "palette" else colorx.bg.rgb_truecolor
        return cls(
            outer_bg_progress=color_fn(*outer_bg_progress),
            outer_bg_1=color_fn(*outer_bg_1),
            outer_bg_2=color_fn(*outer_bg_2),
            outer_bg_3=color_fn(*outer_bg_3),
            outer_bg_4=color_fn(*outer_bg_4),
            board_bg=color_fn(*board_bg),
            board_bg_alt=color_fn(*board_bg_alt),
            board_fg=color_fn(*board_fg),
            empty=color_fn(*empty),
        )

    @classmethod
    def index_of_color(
        cls,
        color_name: Literal[
            "outer_bg_progress",
            "outer_bg_1",
            "outer_bg_2",
            "outer_bg_3",
            "outer_bg_4",
            "board_bg",
            "board_bg_alt",
            "board_fg",
            "empty",
        ],
    ) -> int:
        try:
            return cls._fields.index(color_name)
        except ValueError as e:
            msg = f"Invalid color name: '{color_name}'"
            raise ValueError(msg) from e
