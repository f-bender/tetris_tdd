import numpy as np
from numpy.typing import NDArray

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


def test_colorize() -> None:
    # fmt: off
    space = np.array(
        [
            [ 1,  2,  2,  2,  7,  7,  7,  7, 25, 25],
            [ 1,  3,  2,  6,  6,  6,  6,  8, 24, 25],
            [ 1,  3,  3,  5,  5,  5,  9,  8, 24, 25],
            [ 1,  3,  4,  4,  5, 10,  9,  8, 24, 23],
            [15, 15,  4,  4, 11, 10,  9,  8, 24, 23],
            [15, 14, 14, 11, 11, 10,  9, 21, 21, 23],
            [15, 14, 13, 11, 12, 10, 18, 20, 21, 23],
            [16, 14, 13, 13, 12, 18, 18, 20, 21, 22],
            [16, 17, 17, 13, 12, 18, 20, 20, 22, 22],
            [16, 16, 17, 17, 12, 19, 19, 19, 19, 22],
        ],
        dtype=np.int32,
    )
    # fmt: on
    colored_space = FourColorizer(space).colorize()

    colors = np.unique(colored_space)
    assert set(colors) <= {1, 2, 3, 4}

    for block_index in np.unique(space):
        block_colors = np.unique(colored_space[space == block_index])
        assert len(block_colors) == 1, f"Block {block_index} has multiple colors ({block_colors.tolist()})"

        block_color = block_colors[0]
        assert not any(
            colored_space[neighbor_position] == block_color
            for neighbor_position in _neighboring_positions(space, block_index)
        ), f"Block {block_index} has a neighbor with the same color ({block_color})"


def _neighboring_positions(space: NDArray[np.int32], block_index: int) -> set[tuple[int, int]]:
    block_positions = np.argwhere(space == block_index)
    return {
        tuple(neighbor_position)
        for offset in ((-1, 0), (0, -1), (1, 0), (0, 1))
        for neighbor_position in block_positions + offset
        if neighbor_position not in block_positions
        and (0 <= neighbor_position[0] < space.shape[0] and 0 <= neighbor_position[1] < space.shape[1])
    }
