import inspect
from typing import Literal, get_type_hints

from tetris.ui.cli.color_palette import ColorPalette, _Colors


def test_color_palette_from_rgb_parameters() -> None:
    fields = _Colors._fields

    from_rgb_parameters = inspect.signature(ColorPalette.from_rgb).parameters
    assert set(from_rgb_parameters) == set(fields) | {
        "dynamic_colormap_background",
        "dynamic_colormap_powerup",
        "color_fn",
    }


def test_color_palette_index_of_color_type_hint() -> None:
    fields = _Colors._fields

    color_name_type_hint = get_type_hints(ColorPalette.index_of_color)["color_name"]
    assert color_name_type_hint.__origin__ == Literal
    assert set(color_name_type_hint.__args__) == set(fields)
