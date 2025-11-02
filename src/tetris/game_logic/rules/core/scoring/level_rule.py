from typing import NamedTuple

from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
from tetris.game_logic.rules.messages import NewLevelMessage, NumClearedLinesMessage


class LevelTracker(Publisher, Subscriber):
    def __init__(self, line_clears_per_level: int = 10) -> None:
        super().__init__()

        self._level = 0
        self._line_clears_per_level = line_clears_per_level

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, ClearedLinesTracker) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, NumClearedLinesMessage):
            new_level = message.num_cleared_lines // self._line_clears_per_level
            if new_level != self._level:
                self._level = new_level
                self.notify_subscribers(NewLevelMessage(level=self._level))
