from collections.abc import Callable
from unittest.mock import Mock

import pytest

from tetris.game_logic.components.board import Board
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule


@pytest.fixture(autouse=True)
def clear_full_lines_fn() -> Callable[[Board], None]:
    return lambda board: ClearFullLinesRule().apply(board=board, frame_counter=Mock(), action_counter=Mock())


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
