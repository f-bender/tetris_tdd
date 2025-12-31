import random
from abc import ABC
from functools import cache, lru_cache
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from tetris.ui.cli.color_palette import ColorPalette
from tetris.ui.cli.vec import Vec


class OverlayAnimation(ABC):
    # 0 stands for transparency, other values for color palette indices
    # Note: this means color of index 0 can't be in the animation. This is a limitation we accept for now.
    _FRAMES: ClassVar[NDArray[np.uint8]]  # dimensions: (frame, height, width)

    @classmethod
    def get_frame(cls, current_frame: int, total_frames: int) -> NDArray[np.uint8]:
        return cls._FRAMES[int(current_frame / total_frames * len(cls._FRAMES))]


class TetrisAnimationLeft(OverlayAnimation):
    _FRAMES = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
        ],
        dtype=np.uint8,
    ) * np.uint8(ColorPalette.index_of_color("tetris_sparkle"))

    # offset from the left side of the board at the height of the top cleared tetris row to the intended root of the
    # animation
    OFFSET = Vec(-4, -5)


class TetrisAnimationRight(OverlayAnimation):
    _FRAMES = TetrisAnimationLeft._FRAMES[..., ::-1]  # noqa: SLF001

    # offset from the right side of the board at the height of the top cleared tetris row to the intended root of the
    # animation
    OFFSET = Vec(-4, 0)


class PowerupTriggeredAnimation(OverlayAnimation):
    _FRAMES = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.uint8,
    ) * np.uint8(ColorPalette.DYNAMIC_POWERUP_INDEX)

    # offset from the left side of the board at the height of the top cleared tetris row to the intended root of the
    # animation
    OFFSET = Vec(-2, -5)


@lru_cache
def generate_blooper_overlay(size: tuple[int, int], seed: int) -> NDArray[np.uint8]:
    overlay = np.zeros(size, dtype=np.uint8)

    shorter_side, longer_side = min(size), max(size)
    if shorter_side < 3:  # noqa: PLR2004
        msg = "Boards skinnier than 3 pixels are currently not supported for blooper overlays"
        raise ValueError(msg)

    rng = random.Random(seed)

    min_num_smudges = 4
    # make sure a very elongated overlay *could* still contain enough smudges to basically fill it up
    max_num_smudges = max(longer_side // shorter_side, 7)
    num_smudges = rng.randint(min_num_smudges, max_num_smudges)

    max_radius = max(1, shorter_side // 3)
    min_radius = min(3, max_radius)

    y_coords, x_coords = np.ogrid[: size[0], : size[1]]

    for _ in range(num_smudges):
        radius = rng.randint(min_radius, max_radius)

        y_center = rng.randrange(radius, size[0] - radius)
        x_center = rng.randrange(radius, size[1] - radius)

        # Note: without the "+ 0.5", there would be single-pixel "spikes" in all cardinal directions
        overlay[(y_coords - y_center) ** 2 + (x_coords - x_center) ** 2 <= (radius + 0.5) ** 2] = (
            ColorPalette.index_of_color("blooper_overlay")
        )

    return overlay


class BlooperAnimation:
    _OVERHANG = 3
    _HEIGHT_RATIO = 0.6

    def __init__(self, board_size: tuple[int, int], seed: int) -> None:
        base_overlay_height = int(board_size[0] * self._HEIGHT_RATIO)

        self._overlay = generate_blooper_overlay(
            size=(
                base_overlay_height + self._OVERHANG * 2,
                board_size[1] + self._OVERHANG * 2,
            ),
            seed=seed,
        )
        # without the (empirically adjusted) "+ 4", it disappears too abruptly
        self._final_y_offset = board_size[0] - base_overlay_height + 4

    def get_offset_and_frame(self, current_frame: int, total_frames: int) -> tuple[Vec, NDArray[np.uint8]]:
        offset_from_board_origin = Vec(-self._OVERHANG, -self._OVERHANG)

        # slide down, starting slowly and speeding up (quadratically)
        y_offset = round(self._final_y_offset * (current_frame / total_frames) ** 2)
        offset_from_board_origin += Vec(y_offset, 0)

        return offset_from_board_origin, self._overlay


@cache
def generate_screen_hide_overlay(
    current_frame: int, total_frames: int, screen_size: tuple[int, int]
) -> NDArray[np.uint8]:
    overlay = np.zeros(screen_size, dtype=np.uint8)

    global_fill_ratio = (current_frame + 1) / total_frames

    for y in range(screen_size[0]):
        # first line fills up from global_fill_ratio values 0 to 0.5 (is completely full when global_fill_ratio > 0.5)
        # last line fills up from global_fill_ratio values 0.5 to 1 (is completely empty when global_fill_ratio < 0.5)
        # in between, these limits change linearly
        line_specific_fill_ratio = max(0, min(1, (global_fill_ratio - y / screen_size[0] / 2) * 2))
        overlay[y, : round(screen_size[1] * line_specific_fill_ratio**2)] = ColorPalette.index_of_color(
            "screen_wipe_overlay"
        )

    return overlay


def generate_screen_reveal_overlay(
    current_frame: int, total_frames: int, screen_size: tuple[int, int]
) -> NDArray[np.uint8]:
    return generate_screen_hide_overlay(total_frames - current_frame - 1, total_frames, screen_size)[::-1, ::-1]
