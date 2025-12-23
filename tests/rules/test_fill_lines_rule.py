from collections.abc import Callable

import pytest

from tetris.game_logic.components.board import Board
from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines


@pytest.fixture(autouse=True)
def clear_full_lines_fn() -> Callable[[Board], None]:
    return lambda board: FillLines.clear_full_lines().manipulate_gradually(board=board, current_frame=0, total_frames=1)


def test_one_line_cleared(clear_full_lines_fn: Callable[[Board], None]) -> None:
    board = Board.from_string_representation(
        """
            ..........
            XXXXXXXXXX
        """,
    )

    clear_full_lines_fn(board)
    assert (
        str(board)
        == """
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_four_lines_cleared(clear_full_lines_fn: Callable[[Board], None]) -> None:
    board = Board.from_string_representation(
        """
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
        """,
    )

    clear_full_lines_fn(board)
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_lines_above_clear_drop_down(clear_full_lines_fn: Callable[[Board], None]) -> None:
    board = Board.from_string_representation(
        """
            X........X
            X..XX....X
            XXXXXXXXXX
            X....XXXXX
        """,
    )

    clear_full_lines_fn(board)


"""
            ..........
            X........X
            X..XX....X
            X....XXXXX
        """.replace(" ", "").strip()


def test_lines_above_disconnected_line_clear_drop_down_correctly(clear_full_lines_fn: Callable[[Board], None]) -> None:
    board = Board.from_string_representation(
        """
            X........X
            XXXXXXXXXX
            X....XXXXX
            XXXXXXXXXX
        """,
    )

    clear_full_lines_fn(board)
    assert (
        str(board)
        == """
            ..........
            ..........
            X........X
            X....XXXXX
        """.replace(" ", "").strip()
    )
