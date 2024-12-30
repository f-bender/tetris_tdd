from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.rule import Publisher


class LineClearMessage(NamedTuple):
    cleared_lines: list[int]


class ClearFullLinesRule(Publisher):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        full_lines = board.get_full_line_idxs()
        if not full_lines:
            return

        board.clear_lines(full_lines)
        message = LineClearMessage(cleared_lines=full_lines)
        self.notify_subscribers(message)
