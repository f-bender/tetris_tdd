from math import ceil, floor
from typing import NamedTuple, override

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
from tetris.game_logic.rules.messages import FinishedLineClearMessage, StartingLineClearMessage


class ClearFullLines(Callback, Publisher, Subscriber, GradualBoardManipulation):
    def __init__(self) -> None:
        super().__init__()
        self._full_lines: list[int] | None = None

    @override
    def add_subscriber(self, subscriber: Subscriber) -> None:
        # make sure the ScoreTracker (if it exists) is the first subscriber being notified, to ensure scoring happens
        # before potential levelup
        if isinstance(subscriber, ScoreTracker):
            self._subscribers.insert(0, subscriber)
        else:
            self._subscribers.append(subscriber)

    def manipulate_gradually(self, board: Board, current_frame: int, total_frames: int) -> None:
        assert not board.has_active_block(), "manipulate_gradually was called with an active block on the board!"

        if current_frame == 0:
            assert self._full_lines is None, (
                "manipulate_gradually was called with current_frame = 0 without the preceding manipulation having been "
                "finished (through a call with current_frame = total_frames - 1)!"
            )

            self._full_lines = board.get_full_line_idxs()
            if self._full_lines:
                self.notify_subscribers(
                    StartingLineClearMessage(cleared_lines=self._full_lines, num_frames=total_frames)
                )

        assert self._full_lines is not None, (
            f"manipulate_gradually was called with {current_frame = } without a preceding call with current_frame = 0!"
        )

        if self._full_lines:
            if current_frame == total_frames - 1:
                board.clear_lines(self._full_lines)
                self.notify_subscribers(FinishedLineClearMessage(cleared_lines=self._full_lines))
                self._full_lines = None
            else:
                self._clear_lines_partially(board, portion=(current_frame + 1) / total_frames)
        elif current_frame == total_frames - 1:
            self._full_lines = None

    def _clear_lines_partially(self, board: Board, portion: float) -> None:
        """Clear a portion of the full lines, starting from the middle."""
        assert self._full_lines is not None

        num_cells_to_clear_per_side = round(ceil(board.width / 2) * portion)
        start_index = ceil(board.width / 2) - num_cells_to_clear_per_side
        end_index = floor(board.width / 2) + num_cells_to_clear_per_side

        if start_index < end_index:
            board.array_view_without_active_block()[self._full_lines, start_index:end_index] = 0

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index

    def on_game_start(self) -> None:
        self._full_lines = None

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule

        return isinstance(publisher, Tetris99Rule) and publisher.game_index == self.game_index

    def notify(self, message: NamedTuple) -> None:
        from tetris.game_logic.rules.messages import BoardTranslationMessage

        if self._full_lines and isinstance(message, BoardTranslationMessage):
            self._full_lines = [new_line_idx for i in self._full_lines if (new_line_idx := i + message.y_offset) >= 0]
