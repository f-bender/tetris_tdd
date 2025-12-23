from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.messages import FinishedLineFillMessage, NumClearedLinesMessage


class ClearedLinesTracker(Callback, Publisher, Subscriber):
    def __init__(self) -> None:
        super().__init__()

        self._num_cleared_lines = 0
        self._session_max_cleared_lines = 0

    @property
    def num_cleared_lines(self) -> int:
        return self._num_cleared_lines

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index  # own game: for on_game_start

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines

        return (
            isinstance(publisher, FillLines) and publisher.is_line_clearer and publisher.game_index == self.game_index
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def on_game_start(self) -> None:
        self._num_cleared_lines = 0
        self.notify_subscribers(
            NumClearedLinesMessage(
                num_cleared_lines=self._num_cleared_lines, session_max_cleared_lines=self._session_max_cleared_lines
            )
        )

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, FinishedLineFillMessage):
            self._num_cleared_lines += len(message.filled_lines)
            self._session_max_cleared_lines = max(self._session_max_cleared_lines, self._num_cleared_lines)
            self.notify_subscribers(
                NumClearedLinesMessage(
                    num_cleared_lines=self._num_cleared_lines, session_max_cleared_lines=self._session_max_cleared_lines
                )
            )
