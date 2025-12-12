import logging
import random
from typing import override

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.board import Board
from tetris.game_logic.rules.board_manipulations.board_manipulation import BoardManipulation

LOGGER = logging.getLogger(__name__)


class Gravity(BoardManipulation):
    def __init__(self, per_col_probability: float = 1) -> None:
        self._per_col_probability = per_col_probability

    @override
    def manipulate(self, board: Board) -> None:
        if board.has_active_block():
            LOGGER.warning("Gravity application while there is an active block is not supported!")
            return

        board.set_from_array(self._apply_gravity(board.as_array()))

    def _apply_gravity(self, array: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """All falsy values 'float' to the top of each column, all truthy values 'fall' to the bottom of each column."""
        # Create an output array to store the sorted columns
        sorted_arr = np.copy(array)

        for col in range(array.shape[1]):
            # random chance to skip the column
            if random.random() > self._per_col_probability:
                continue

            # Get the column
            column = array[:, col]

            # Create a mask for falsy values (zeros)
            falsy_mask = column == 0

            # Get the falsy and truthy parts
            falsy_part = column[falsy_mask]
            truthy_part = column[~falsy_mask]

            # Combine falsy and truthy parts, falsy first
            sorted_column = np.concatenate([falsy_part, truthy_part])

            # Assign the sorted column back to the output array
            sorted_arr[:, col] = sorted_column

        return sorted_arr
