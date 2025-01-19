from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule, LineClearMessage


class TrackScoreCallback(Callback, Subscriber):
    def __init__(self, header: str | None = None) -> None:
        super().__init__()

        self._score = 0
        self._high_score = 0
        self._header = header

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index in (
            -1,  # runtime: for on_frame_start
            self.game_index,  # own game: for on_game_start
        )

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, ClearFullLinesRule) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

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
