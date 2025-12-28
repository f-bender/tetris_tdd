from collections.abc import Callable
from math import ceil, floor
from typing import NamedTuple, Self, override

import numpy as np

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
from tetris.game_logic.rules.messages import FinishedLineFillMessage, StartingLineFillMessage


class FillLines(Callback, Publisher, Subscriber, GradualBoardManipulation):
    def __init__(self, line_idx_factory: Callable[[Board], list[int]], fill_value: int) -> None:
        super().__init__()

        self._line_idx_factory = line_idx_factory
        self._fill_value = np.uint8(fill_value)
        self._lines_to_fill: list[int] | None = None

    @classmethod
    def clear_full_lines(cls) -> Self:
        return cls(line_idx_factory=Board.get_full_line_idxs, fill_value=0)

    @property
    def fill_value(self) -> np.uint8:
        return self._fill_value

    @property
    def line_idx_factory(self) -> Callable[[Board], list[int]]:
        return self._line_idx_factory

    @line_idx_factory.setter
    def line_idx_factory(self, factory: Callable[[Board], list[int]]) -> None:
        self._line_idx_factory = factory

    @property
    def is_line_clearer(self) -> bool:
        return self._fill_value == 0

    @override
    def add_subscriber(self, subscriber: Subscriber) -> None:
        if not self.is_line_clearer:
            super().add_subscriber(subscriber)
            return

        # make sure the ScoreTracker (if it exists) is the first subscriber being notified, to ensure scoring happens
        # before potential levelup
        if isinstance(subscriber, ScoreTracker):
            self._subscribers.insert(0, subscriber)
        else:
            self._subscribers.append(subscriber)

    @override
    def done_already(self) -> bool:
        return not self._lines_to_fill

    @override
    def manipulate_gradually(self, board: Board, current_frame: int, total_frames: int) -> None:
        assert not board.has_active_block(), "manipulate_gradually was called with an active block on the board!"

        if current_frame == 0:
            self._lines_to_fill = self._line_idx_factory(board)
            if self._lines_to_fill:
                self.notify_subscribers(
                    StartingLineFillMessage(
                        filled_lines=self._lines_to_fill, is_line_clear=self.is_line_clearer, num_frames=total_frames
                    )
                )

        assert self._lines_to_fill is not None, (
            f"manipulate_gradually was called with {current_frame = } without a preceding call with current_frame = 0!"
        )

        if self._lines_to_fill:
            if current_frame == total_frames - 1:
                if self.is_line_clearer:
                    board.clear_lines(self._lines_to_fill)
                self.notify_subscribers(
                    FinishedLineFillMessage(filled_lines=self._lines_to_fill, is_line_clear=self.is_line_clearer)
                )
                self._lines_to_fill = None
            else:
                self._fill_lines_partially(board, portion=(current_frame + 1) / total_frames)
        elif current_frame == total_frames - 1:
            self._lines_to_fill = None

    def _fill_lines_partially(self, board: Board, portion: float) -> None:
        """Fill a portion of the lines with the fill value, starting from the middle."""
        assert self._lines_to_fill is not None

        num_cells_to_fill_per_side = round(ceil(board.width / 2) * portion)
        start_index = ceil(board.width / 2) - num_cells_to_fill_per_side
        end_index = floor(board.width / 2) + num_cells_to_fill_per_side

        if start_index < end_index:
            board.array_view_without_active_block()[self._lines_to_fill, start_index:end_index] = self._fill_value

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index

    def on_game_start(self, game_index: int) -> None:
        self._lines_to_fill = None

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule

        return isinstance(publisher, Tetris99Rule) and publisher.game_index == self.game_index

    def notify(self, message: NamedTuple) -> None:
        from tetris.game_logic.rules.messages import BoardTranslationMessage

        if self._lines_to_fill and isinstance(message, BoardTranslationMessage):
            self._lines_to_fill = [
                new_line_idx for i in self._lines_to_fill if (new_line_idx := i + message.y_offset) >= 0
            ]
