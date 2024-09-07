from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection


class LineClearMessage(NamedTuple):
    cleared_lines: list[int]


class ClearFullLinesRule:
    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        full_lines = board.get_full_line_idxs()
        board.clear_lines(full_lines)
        callback_collection.custom_message(LineClearMessage(cleared_lines=full_lines))
