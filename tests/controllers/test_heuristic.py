import numpy as np

from tetris.controllers.heuristic_bot.heuristic import Heuristic


def test_sum_cell_heights_close_to_top() -> None:
    board_array = np.array(
        [
            [0, 1, 0],  # 1 cell, height 3
            [1, 1, 0],  # 2 cells, height 2
            [1, 1, 1],  # 3 cells, height 1
        ],
        dtype=bool,
    )

    # Test with threshold 2 (only top 2 rows)
    result = Heuristic.sum_cell_heights_close_to_top(board_array, 2)
    assert result == 4  # (2 * 1) + (1 * 2) = 4 total height units  # noqa: PLR2004


def test_sum_cell_heights_close_to_top_empty() -> None:
    board_array = np.zeros((3, 3), dtype=bool)
    result = Heuristic.sum_cell_heights_close_to_top(board_array, 2)
    assert result == 0


def test_count_rows_with_overhung_holes_simple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],  # overhung row
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.count_rows_with_overhung_cells(board_array)
    assert result == 1


def test_count_rows_with_overhung_holes_multiple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],  # overhung row 1
            [1, 0, 1],  # overhung row 2
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.count_rows_with_overhung_cells(board_array)
    assert result == 2  # noqa: PLR2004


def test_count_distinct_overhangs_simple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],  # one overhang above
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.count_distinct_overhangs(board_array)
    assert result == 1


def test_count_distinct_overhangs_multiple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],  # first overhang
            [1, 1, 0],
            [1, 0, 0],  # second overhang
        ],
        dtype=bool,
    )

    result = Heuristic.count_distinct_overhangs(board_array)
    assert result == 2  # noqa: PLR2004


def test_count_overhanging_cells_simple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],  # 1 overhanging
            [0, 0, 0],  # 1 overhung
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.count_overhanging_cells(board_array)
    assert result == 1


def test_count_overhanging_cells_complex() -> None:
    board_array = np.array(
        [
            [0, 1, 1],  # 2 overhanging
            [0, 0, 0],  # 2 overhung
            [1, 1, 0],  # 1 overhanging, 1 overhung
            [1, 0, 0],  # 2 overhung
        ],
        dtype=bool,
    )

    result = Heuristic.count_overhanging_cells(board_array)
    assert result == 3  # noqa: PLR2004


def test_count_overhung_cells_simple() -> None:
    board_array = np.array(
        [
            [0, 1, 0],  # 1 overhanging
            [0, 0, 0],  # 1 overhung
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.count_overhung_cells(board_array)
    assert result == 1


def test_count_overhung_cells_complex() -> None:
    board_array = np.array(
        [
            [0, 1, 1],  # 2 overhanging
            [0, 0, 0],  # 2 overhung
            [1, 1, 0],  # 1 overhanging, 1 overhung
            [1, 0, 0],  # 2 overhung
        ],
        dtype=bool,
    )

    result = Heuristic.count_overhung_cells(board_array)
    assert result == 5  # noqa: PLR2004


def test_count_narrow_gaps_none() -> None:
    height_diffs = np.array([-2, 2])  # Below threshold
    result = Heuristic.count_narrow_gaps(height_diffs)
    assert result == 0


def test_count_narrow_gaps_middle() -> None:
    height_diffs = np.array([-3, 3])  # One gap in middle
    result = Heuristic.count_narrow_gaps(height_diffs)
    assert result == 1


def test_count_narrow_gaps_edges() -> None:
    # Gaps at both edges
    height_diffs = np.array([3, -3])
    result = Heuristic.count_narrow_gaps(height_diffs)
    assert result == 2  # noqa: PLR2004


def test_sum_cell_heights_simple() -> None:
    board_array = np.array(
        [
            [0, 0, 0],  # height 3
            [0, 1, 0],  # one at height 2
            [1, 1, 1],  # three at height 1
        ],
        dtype=bool,
    )

    result = Heuristic.sum_cell_heights(board_array)
    assert result == 5  # (1 * 2) + (3 * 1) = 5  # noqa: PLR2004


def test_sum_cell_heights_empty() -> None:
    board_array = np.zeros((3, 3), dtype=bool)
    result = Heuristic.sum_cell_heights(board_array)
    assert result == 0


def test_adjacent_height_differences_flat() -> None:
    board_array = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.adjacent_height_differences(board_array)
    np.testing.assert_array_equal(result, [0, 0])


def test_adjacent_height_differences_stairs() -> None:
    board_array = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result = Heuristic.adjacent_height_differences(board_array)
    np.testing.assert_array_equal(result, [1, 1])


def test_adjacent_height_differences_empty_columns() -> None:
    board_array = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    )

    result = Heuristic.adjacent_height_differences(board_array)
    np.testing.assert_array_equal(result, [3, -3])
