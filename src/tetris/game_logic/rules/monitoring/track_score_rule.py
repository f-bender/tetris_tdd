from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.messages import FinishedLineClearMessage


class ScoreMessage(NamedTuple):
    score: int
    high_score: int


class ScoreTracker(Callback, Publisher, Subscriber):
    def __init__(self) -> None:
        super().__init__()

        self._score = 0
        self._high_score = 0

    @property
    def score(self) -> int:
        return self._score

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index  # own game: for on_game_start

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, ClearFullLines) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def on_game_start(self) -> None:
        self._score = 0
        self.notify_subscribers(ScoreMessage(score=self._score, high_score=self._high_score))

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, FinishedLineClearMessage):
            self._score += len(message.cleared_lines)
            self._high_score = max(self._high_score, self._score)
            self.notify_subscribers(ScoreMessage(score=self._score, high_score=self._high_score))
