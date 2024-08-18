import numpy as np
import pytest
from numpy.typing import NDArray

from ui.cli.tetromino_space_filler import TetrominoSpaceFiller


@pytest.mark.parametrize(
    "space, check_around, expected",
    [
        (
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            None,
            True,
        ),
        (
            np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                dtype=np.int16,
            ),
            None,
            True,
        ),
        (
            np.array(
                [
                    [0, -1, 0, 0],
                    [0, -1, -1, 0],
                    [0, 0, -1, 0],
                ],
                dtype=np.int16,
            ),
            None,
            True,
        ),
        (
            # badly placed block splits space into non-divisible-by-4 islands
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                dtype=np.int16,
            ),
            None,
            False,
        ),
        (
            # array has spaces not divisible by 4
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int16,
            ),
            None,
            False,
        ),
        (
            # hole splits area into non-divisible-by-4 islands
            np.array(
                [
                    [0, -1, 0],
                    [0, -1, 0],
                    [-1, -1, 0],
                    [0, 0, 0],
                ],
                dtype=np.int16,
            ),
            None,
            False,
        ),
    ],
)
def test_space_fillable(space: NDArray[np.int16], check_around: tuple[slice, slice] | None, expected: bool) -> None:
    assert TetrominoSpaceFiller.space_fillable(space_view=space, check_around=check_around) is expected
