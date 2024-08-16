from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.callback_collection import CallbackCollection


class ClearFullLinesRule:
    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        full_lines = board.get_full_line_idxs()
        board.clear_lines(full_lines)
        callback_collection.custom_message(f"line_clear {full_lines}")
