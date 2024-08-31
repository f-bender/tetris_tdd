import contextlib

import numpy as np
import pytest
from numpy.typing import NDArray

from ui.cli.tetromino_space_filler import TetrominoSpaceFiller


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        (
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            np.array(
                [
                    [-1, -1, 0, 0],
                    [-1, -1, 0, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            np.array(
                [
                    [0, -1, 0, 0],
                    [0, -1, -1, 0],
                    [0, 0, -1, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            # badly placed block splits space into non-divisible-by-4 islands
            np.array(
                [
                    [0, -1, 0, 0],
                    [-1, -1, -1, 0],
                ],
                dtype=np.int32,
            ),
            False,
        ),
        (
            # array has spaces not divisible by 4
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int32,
            ),
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
                dtype=np.int32,
            ),
            False,
        ),
    ],
)
def test_space_fillable(space: NDArray[np.int32], expected: bool) -> None:
    with (
        pytest.raises(
            ValueError, match="Space cannot be filled! Contains at least one island with size not divisible by 4!"
        )
        if expected is False
        else contextlib.nullcontext()
    ):
        TetrominoSpaceFiller(space)
