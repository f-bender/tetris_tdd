import random
from typing import NamedTuple

import numpy as np

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.messages import BoardTranslationMessage, FinishedLineClearMessage, Tetris99Message


class PlaceLinesManipulation:
    def __init__(self, num_lines: int) -> None:
        self._num_lines = num_lines

    def manipulate(self, board: Board) -> None:
        board_array = board.array_view_without_active_block().copy()

        if np.any(board_array[: self._num_lines]):
            raise GameOverError

        row_to_fill_in = np.ones_like(board_array[0]) * board.NEUTRAL_BLOCK_INDEX
        row_to_fill_in[random.randrange(len(board_array[0]))] = 0

        board_array[: -self._num_lines] = board_array[self._num_lines :]
        board_array[-self._num_lines :] = row_to_fill_in

        board.set_from_array(board_array, active_block_displacement=(-self._num_lines, 0))


class Tetris99Rule(Publisher, Subscriber):
    def __init__(self, target_idxs: list[int], targeted_by_idxs: list[int] | None = None) -> None:
        super().__init__()

        if not target_idxs:
            msg = "At least one target_id must be provided."
            raise ValueError(msg)

        self._target_idxs = target_idxs
        self._targeted_by_idxs = targeted_by_idxs or target_idxs

        self._num_recently_cleared_lines: int = 0
        self._num_lines_to_place: int = 0

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return (isinstance(publisher, ClearFullLines) and publisher.game_index == self.game_index) or (
            isinstance(publisher, Tetris99Rule) and publisher.game_index in self._targeted_by_idxs
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        num_clear_line_subscriptions = sum(1 for publisher in publishers if isinstance(publisher, ClearFullLines))
        num_tetris_99_subscriptions = sum(1 for publisher in publishers if isinstance(publisher, Tetris99Rule))

        if num_clear_line_subscriptions != 1:
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {num_clear_line_subscriptions} "
                "ClearFullLines subscriptions!"
            )
            raise RuntimeError(msg)

        if num_tetris_99_subscriptions != len(self._targeted_by_idxs):
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {num_tetris_99_subscriptions} other Tetris99Rule "
                f"subscriptions, expected {len(self._targeted_by_idxs)}!"
            )
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, FinishedLineClearMessage):
            self._num_recently_cleared_lines = len(message.cleared_lines)

        elif isinstance(message, Tetris99Message) and message.target_id == self.game_index:
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
                    target_id=random.choice(self._target_idxs),
                )
            )
            self._num_recently_cleared_lines = 0

        if self._num_lines_to_place:
            PlaceLinesManipulation(self._num_lines_to_place).manipulate(board)
            self.notify_subscribers(BoardTranslationMessage(y_offset=-self._num_lines_to_place))
            self._num_lines_to_place = 0
