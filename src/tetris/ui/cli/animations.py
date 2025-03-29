from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from tetris.ui.cli.color_palette import ColorPalette
from tetris.ui.cli.vec import Vec


@dataclass(slots=True)
class Overlay:
    position: Vec
    frame: NDArray[np.uint8]

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]


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
