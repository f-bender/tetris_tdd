import random
from functools import cache
from typing import override

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.messages import GravityFinishedMessage, GravityStartedMessage


class Gravity(GradualBoardManipulation, Publisher):
    def __init__(self, per_col_probability: float = 1) -> None:
        super().__init__()

        self._per_col_probability = per_col_probability
        self._currently_affected_columns: list[int] | None = None
        self._currently_total_steps: int | None = None
        self._done_already = False

    @property
    def per_col_probability(self) -> float:
        return self._per_col_probability

    @per_col_probability.setter
    def per_col_probability(self, value: float) -> None:
        self._per_col_probability = value

    @override
    def done_already(self) -> bool:
        return self._done_already

    @override
    def manipulate_gradually(self, board: Board, current_frame: int, total_frames: int) -> None:
        assert not board.has_active_block(), "manipulate_gradually was called with an active block on the board!"

        if current_frame == 0:
            self._done_already = False
            self._currently_affected_columns = [
                i for i in range(board.width) if random.random() < self._per_col_probability
            ]
            self._currently_total_steps = self._estimate_total_num_bubble_steps(board)
            if self._currently_total_steps == 0:
                self._done_already = True
                return

            self.notify_subscribers(GravityStartedMessage())

        assert self._currently_affected_columns is not None, (
            f"manipulate_gradually was called with {current_frame = } without a preceding call with current_frame = 0!"
        )

        num_steps = self._split_steps_across_frames(steps=self._currently_total_steps, frames=total_frames)[
            current_frame
        ]
        if num_steps == 0:
            return

        board_before = board.as_array()
        board_after = board_before

        for _ in range(num_steps):
            board_after = self._bubble_falsy_up(board_after)

        if np.array_equal(board_before, board_after):
            self._done_already = True
        else:
            board.set_from_array(board_after)

        if current_frame == total_frames - 1 or self._done_already:
            self.notify_subscribers(GravityFinishedMessage())

    @staticmethod
    def _estimate_total_num_bubble_steps(board: Board) -> int:
        board_array = board.array_view_without_active_block().view(bool)

        highest_filled_cells = np.where(board_array.any(axis=0), board_array.argmax(axis=0), -1)

        lowest_empty_cells = np.where(
            (~board_array).any(axis=0), board.height - 1 - np.flip(~board_array, axis=0).argmax(axis=0), -1
        )

        valid_idxs = (highest_filled_cells != -1) & (lowest_empty_cells != -1)
        if not np.any(valid_idxs):
            return 0

        return max(0, (lowest_empty_cells[valid_idxs] - highest_filled_cells[valid_idxs]).max())

    def _bubble_falsy_up(self, array: NDArray[np.uint8]) -> NDArray[np.uint8]:
        assert self._currently_affected_columns is not None

        array = array.copy()

        above = array[:-1].view(bool)
        below = array[1:].view(bool)

        for y, x in zip(*np.nonzero((~below) & above), strict=True):
            if x not in self._currently_affected_columns:
                continue

            array[y, x], array[y + 1, x] = array[y + 1, x], array[y, x]

        return array

    @staticmethod
    @cache
    def _split_steps_across_frames(steps: int, frames: int) -> list[int]:
        min_steps, remainder = divmod(steps, frames)
        step_per_frames = [min_steps] * frames

        remainder_idxs = np.round(np.linspace(0, frames - 1, remainder)).astype(int)
        for i in remainder_idxs:
            step_per_frames[i] += 1

        return step_per_frames
