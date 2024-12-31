from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.rule import Subscriber
from tetris.rules.core.clear_full_lines_rule import LineClearMessage


class TrackScoreCallback(Callback, Subscriber):
    def __init__(self, header: str | None = None) -> None:
        self._score = 0
        self._high_score = 0
        self._header = header

    def on_game_start(self) -> None:
        self._score = 0

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, LineClearMessage):
            self._score += len(message.cleared_lines)
            self._high_score = max(self._high_score, self._score)

    def on_frame_start(self) -> None:
        if self._header:
            print(self._header)  # noqa: T201
        print(f"Score: {self._score}")  # noqa: T201
        print(f"High Score: {self._high_score}")  # noqa: T201
        print()  # noqa: T201
