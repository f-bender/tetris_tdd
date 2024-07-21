import numpy as np
from board_manipulations.gravity import Gravity


def test_gravity_manipulation_01() -> None:
    array_before = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    array_after = Gravity()._apply_gravity(array_before)
    assert np.array_equal(
        array_after,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
            ]
        ),
    )


def test_gravity_manipulation_integers() -> None:
    array_before = np.array(
        [
            [3, 7, 2, 0],
            [9, 0, 1, -1],
            [0, 6, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    array_after = Gravity()._apply_gravity(array_before)
    assert np.array_equal(
        array_after,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 7, 2, -1],
                [9, 6, 1, 1],
            ]
        ),
    )
