import contextlib

import numpy as np
import pytest
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block, BlockType
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        (
            -np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            -np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            -np.array(
                [
                    [0, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                ],
                dtype=np.int32,
            ),
            True,
        ),
        (
            # badly placed block splits space into non-divisible-by-4 islands
            -np.array(
                [
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                dtype=np.int32,
            ),
            False,
        ),
        (
            # array has spaces not divisible by 4
            -np.array(
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
            -np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0],
                ],
                dtype=np.int32,
            ),
            False,
        ),
    ],
)
def test_space_fillable(space: NDArray[np.int32], *, expected: bool) -> None:
    with (
        pytest.raises(
            ValueError,
            match="Space cannot be filled! Contains at least one island with size not divisible by 4!",
        )
        if expected is False
        else contextlib.nullcontext()
    ):
        TetrominoSpaceFiller(space)


def array_hash(array: NDArray[np.bool]) -> int:
    return hash(array.tobytes() + bytes(array.shape))


def test_get_unique_rotations_transposes_for_L() -> None:  # noqa: N802
    unique_views = TetrominoSpaceFiller._get_unique_rotations_transposes(Block(BlockType.L).actual_cells.astype(bool))
    assert len(unique_views) == 8  # noqa: PLR2004
    assert {array_hash(view) for view in unique_views} == {
        array_hash(np.array(view, dtype=bool))
        for view in (
            [
                [1, 0],
                [1, 0],
                [1, 1],
            ],
            [
                [1, 1, 1],
                [1, 0, 0],
            ],
            [
                [1, 1],
                [0, 1],
                [0, 1],
            ],
            [
                [0, 0, 1],
                [1, 1, 1],
            ],
            [
                [1, 1],
                [1, 0],
                [1, 0],
            ],
            [
                [1, 1, 1],
                [0, 0, 1],
            ],
            [
                [0, 1],
                [0, 1],
                [1, 1],
            ],
            [
                [1, 0, 0],
                [1, 1, 1],
            ],
        )
    }


def test_get_unique_rotations_transposes_for_O() -> None:  # noqa: N802
    unique_views = TetrominoSpaceFiller._get_unique_rotations_transposes(Block(BlockType.O).actual_cells.astype(bool))
    assert len(unique_views) == 1
    assert {array_hash(view) for view in unique_views} == {array_hash(np.array([[1, 1], [1, 1]], dtype=bool))}


def test_get_unique_rotations_transposes_for_I() -> None:  # noqa: N802
    unique_views = TetrominoSpaceFiller._get_unique_rotations_transposes(Block(BlockType.I).actual_cells.astype(bool))
    assert len(unique_views) == 2  # noqa: PLR2004
    assert {array_hash(view) for view in unique_views} == {
        array_hash(np.array([[1, 1, 1, 1]], dtype=bool)),
        array_hash(np.array([[1], [1], [1], [1]], dtype=bool)),
    }


def test_closest_position_in_area() -> None:
    allowed_area = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )

    # Test when position is in allowed area
    position = (0, 0)
    result = TetrominoSpaceFiller._closest_position_in_area(allowed_area, position)
    assert result == (0, 0)

    # Test when position is not in allowed area
    position = (1, 2)
    result = TetrominoSpaceFiller._closest_position_in_area(allowed_area, position)
    assert result == (1, 0)


def test_check_islands_are_fillable_and_set_smallest_island() -> None:
    space = -np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
    )
    filler = TetrominoSpaceFiller(space)

    result = filler._check_islands_are_fillable_and_set_smallest_island()
    assert result
    assert filler._smallest_island is not None
    assert np.array_equal(
        filler._smallest_island,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
    )


def test_get_neighboring_empty_cell_with_least_empty_neighbors_position_with_empty_position() -> None:
    space = -np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    filler = TetrominoSpaceFiller(space)
    tetromino = Block(BlockType.O).actual_cells
    tetromino_position = (0, 0)

    result = filler._get_neighboring_empty_cell_with_least_empty_neighbors_position(
        tetromino=tetromino,
        tetromino_position=tetromino_position,
    )
    assert result == (2, 1)


def test_get_neighboring_empty_cell_with_least_empty_neighbors_position_without_empty_position() -> None:
    space = -np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    filler = TetrominoSpaceFiller(space)
    tetromino = Block(BlockType.O).actual_cells
    tetromino_position = (0, 0)

    result = filler._get_neighboring_empty_cell_with_least_empty_neighbors_position(
        tetromino=tetromino,
        tetromino_position=tetromino_position,
    )
    assert result is None


def test_count_empty_neighbors() -> None:
    space = -np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    filler = TetrominoSpaceFiller(space)

    assert filler._count_empty_neighbors((1, 1)) == 4  # noqa: PLR2004
    assert filler._count_empty_neighbors((2, 1)) == 3  # noqa: PLR2004
    assert filler._count_empty_neighbors((2, 0)) == 2  # noqa: PLR2004
    assert filler._count_empty_neighbors((3, 2)) == 1
    assert filler._count_empty_neighbors((4, 1)) == 0


def test_fill() -> None:
    height, width = 10, 10
    space = np.zeros((height, width), dtype=np.int32)
    TetrominoSpaceFiller(space).fill()

    TetrominoSpaceFiller.validate_filled_space(space)
