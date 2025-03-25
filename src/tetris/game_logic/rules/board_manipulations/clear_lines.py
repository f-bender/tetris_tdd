from math import ceil, floor

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.messages import FinishedLineClearMessage, StartingLineClearMessage


class ClearFullLines(Publisher, GradualBoardManipulation):
    def __init__(self) -> None:
        super().__init__()
        self._full_lines: list[int] | None = None

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
