import inspect
from typing import Literal, get_type_hints

from ui.cli.color_palette import ColorPalette


def test_color_palette_from_rgb_parameters() -> None:
    fields = ColorPalette._fields

    from_rgb_parameters = inspect.signature(ColorPalette.from_rgb).parameters
    assert set(from_rgb_parameters) == set(fields) | {"mode"}


def test_color_palette_index_of_color_type_hint() -> None:
    fields = ColorPalette._fields

    color_name_type_hint = get_type_hints(ColorPalette.index_of_color)["color_name"]
    assert color_name_type_hint.__origin__ == Literal
    assert set(color_name_type_hint.__args__) == set(fields)
