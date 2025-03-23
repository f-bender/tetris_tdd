from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.rules.core.messages import SpawnMessage
from tetris.rules.monitoring.track_score_rule import ScoreMessage


class UiAggregator(Subscriber):
    """Subscriber to all UI-relevant events, aggregating them into UiElements."""

    def __init__(self, board: NDArray[np.uint8]) -> None:
        super().__init__()
        self._ui_elements = SingleUiElements(board=board)

    def reset(self) -> None:
        self._ui_elements = SingleUiElements(board=self._ui_elements.board)

    @property
    def ui_elements(self) -> SingleUiElements:
        return self._ui_elements

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
        from tetris.rules.monitoring.track_score_rule import TrackScoreRule

        return isinstance(publisher, TrackScoreRule | SpawnStrategyImpl) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 2:  # noqa: PLR2004
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}."
                "Expected 2 subscriptions (TrackScoreRule and SpawnStrategyImpl)."
            )
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        match message:
            case ScoreMessage(score=score):
                self._ui_elements.score = score
            case SpawnMessage(next_block=next_block):
                self._ui_elements.next_block = next_block
            case _:
                msg = f"Unexpected message: {message}"
                raise ValueError(msg)

    def update(self, board: NDArray[np.uint8]) -> None:
        self._ui_elements.board = board
