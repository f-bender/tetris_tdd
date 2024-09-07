from typing import Literal, NamedTuple, Self

from tetris.ansi_extensions import color as colorx


class ColorPalette(NamedTuple):
    board_bg: str
    board_bg_alt: str
    board_fg: str

    @classmethod
    def from_rgb(
        cls,
        board_bg: tuple[int, int, int],
        board_bg_alt: tuple[int, int, int],
        board_fg: tuple[int, int, int],
        mode: Literal["palette", "truecolor"] = "truecolor",
    ) -> Self:
        color_fn = colorx.bg.rgb_palette if mode == "palette" else colorx.bg.rgb_truecolor
        return cls(
            board_bg=color_fn(*board_bg),
            board_bg_alt=color_fn(*board_bg_alt),
            board_fg=color_fn(*board_fg),
        )

    @classmethod
    def index_of_color(
        cls,
        color_name: Literal[
            "board_bg",
            "board_bg_alt",
            "board_fg",
        ],
    ) -> int:
        try:
            return cls._fields.index(color_name)
        except ValueError as e:
            raise ValueError(f"Invalid color name: '{color_name}'") from e
