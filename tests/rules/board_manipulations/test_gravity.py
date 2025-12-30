import pytest

from tetris.game_logic.components.board import Board
from tetris.game_logic.rules.board_manipulations.gravity import Gravity


@pytest.mark.parametrize(
    ("board_string", "expected_steps"),
    [
        (
            """
                X
                .
                .
                .
            """,
            3,
        ),
        (
            """
                ..
                .X
                X.
                ..
            """,
            2,
        ),
        (
            """
                .
                .
                .
                X
            """,
            0,
        ),
        (
            """
                .
                .
                .
                .
            """,
            0,
        ),
        (
            """
                ....
                .X..
                .X.X
                .XXX
            """,
            0,
        ),
        (
            """
                ...X
                .X..
                ...X
                .XXX
            """,
            1,
        ),
        (
            """
                XXXX
                XXXX
                XXXX
                XXXX
            """,
            0,
        ),
        (
            """
                .....
                .XXX.
                XX...
                XXXXX
                XXXX.
                ...XX
            """,
            4,
        ),
    ],
)
def test_estimate_total_num_bubble_steps(board_string: str, expected_steps: int) -> None:
    board = Board.from_string_representation(board_string)

    estimated_steps = Gravity.estimate_total_num_bubble_steps(board)

    assert estimated_steps == expected_steps, f"Expected {expected_steps} steps, got {estimated_steps}. Board:\n{board}"
