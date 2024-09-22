import numpy as np

from tetris.tetromino_space_filler.four_colorizer import FourColorizer


def test_blocks_are_close_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 1, 0],
            [0, 2, 2],
            [0, 0, 0],
        ]
    )
    assert FourColorizer(space)._blocks_are_close(1, 2)
    assert FourColorizer(space, closeness_threshold=1)._blocks_are_close(1, 2)


def test_blocks_are_close_non_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 2, 2],
        ]
    )
    assert FourColorizer(space)._blocks_are_close(1, 2)
    assert not FourColorizer(space, closeness_threshold=1)._blocks_are_close(1, 2)


def test_blocks_are_close_diagonally_adjacent_blocks() -> None:
    space = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ]
    )
    assert FourColorizer(space)._blocks_are_close(1, 2)
    assert not FourColorizer(space, closeness_threshold=1)._blocks_are_close(1, 2)


def test_get_neighboring_colors_and_uncolored_blocks() -> None:
    space = np.array(
        [
            [1, 1, 0, 2],
            [0, 5, 5, 2],
            [3, 3, 4, 4],
        ]
    )
    colorizer = FourColorizer(space)
    colorizer._colored_space = np.array(
        [
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 2, 2],
        ]
    )
    assert colorizer._get_neighboring_colors_and_uncolored_blocks(5) == ({1, 2}, {3})
