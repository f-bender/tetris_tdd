import numpy as np

from tetris.tetromino_space_filler.four_colorizer import FourColorizer


def test_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
            [0, 0, 0],
        ]
    )
    assert FourColorizer._blocks_are_close(space, 1, 2)


def test_non_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 2, 2],
        ]
    )
    assert not FourColorizer._blocks_are_close(space, 1, 2)


def test_diagonally_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ]
    )
    assert not FourColorizer._blocks_are_close(space, 1, 2)
