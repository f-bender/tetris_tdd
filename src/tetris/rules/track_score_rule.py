from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.rules.clear_full_lines_rule import LineClearMessage


class TrackScoreRule(Callback):
    def __init__(self) -> None:
        self._score = 0
        self._high_score = 0

    def on_game_start(self) -> None:
        self._score = 0

    def custom_message(self, message: NamedTuple) -> None:
        if isinstance(message, LineClearMessage):
            self._score += len(message.cleared_lines)
            self._high_score = max(self._high_score, self._score)

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        print(f"Score: {self._score}")
        print(f"High Score: {self._high_score}")
