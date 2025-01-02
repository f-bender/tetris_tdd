import random
from typing import NamedTuple

import numpy as np

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.rule import Publisher, Subscriber
from tetris.rules.core.clear_full_lines_rule import LineClearMessage


class PlaceLinesManipulation:
    NEUTRAL_BLOCK_INDEX = 8

    def __init__(self, num_lines: int) -> None:
        self._num_lines = num_lines

    def manipulate(self, board: Board) -> None:
        board_array = board.as_array_without_active_block()

        if np.any(board_array[: self._num_lines]):
            raise GameOverError

        row_to_fill_in = np.ones_like(board_array[0]) * self.NEUTRAL_BLOCK_INDEX
        row_to_fill_in[random.randrange(len(board_array[0]))] = 0

        board_array[: -self._num_lines] = board_array[self._num_lines :]
        board_array[-self._num_lines :] = row_to_fill_in

        board.set_from_array(board_array, active_block_displacement=(-self._num_lines, 0))


class Tetris99Message(NamedTuple):
    num_lines: int
    target_id: int


class Tetris99Rule(Publisher, Subscriber):
    def __init__(self, id: int, target_ids: list[int]) -> None:  # noqa: A002
        super().__init__()

        if not target_ids:
            msg = "At least one target_id must be provided."
            raise ValueError(msg)

        self._id = id
        self._target_ids = target_ids

        self._num_recently_cleared_lines: int = 0
        self._num_lines_to_place: int = 0

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, LineClearMessage):
            self._num_recently_cleared_lines = len(message.cleared_lines)

        elif isinstance(message, Tetris99Message) and message.target_id == self._id:
            self._num_lines_to_place += message.num_lines

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        if self._num_recently_cleared_lines:
            self.notify_subscribers(
                Tetris99Message(
                    num_lines=self._num_recently_cleared_lines,
                    target_id=random.choice(self._target_ids),
                )
            )
            self._num_recently_cleared_lines = 0

        if self._num_lines_to_place:
            PlaceLinesManipulation(self._num_lines_to_place).manipulate(board)
            self._num_lines_to_place = 0
