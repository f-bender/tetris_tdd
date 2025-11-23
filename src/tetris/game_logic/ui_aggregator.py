from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.interfaces.animations import TetrisAnimationSpec
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.game_logic.rules.messages import (
    FinishedLineClearMessage,
    NewLevelMessage,
    NumClearedLinesMessage,
    PowerupTTLsMessage,
    ScoreMessage,
    SpawnMessage,
    StartingLineClearMessage,
)
from tetris.game_logic.rules.special.powerup import PowerupRule


class UiAggregator(Subscriber):
    """Subscriber to all UI-relevant events, aggregating them into UiElements."""

    def __init__(self, board: NDArray[np.uint8], controller_symbol: str) -> None:
        super().__init__()
        self._ui_elements = SingleUiElements(board=board, controller_symbol=controller_symbol)

    def reset(self) -> None:
        self._ui_elements.reset()

    @property
    def ui_elements(self) -> SingleUiElements:
        return self._ui_elements

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
        from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
        from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
        from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
        from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl

        return (
            isinstance(
                publisher,
                ClearedLinesTracker | SpawnStrategyImpl | ClearFullLines | ScoreTracker | LevelTracker | PowerupRule,
            )
            and publisher.game_index == self.game_index
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
        from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
        from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl

        if not any(isinstance(publisher, ClearedLinesTracker) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to ClearedLinesTracker."
            raise RuntimeError(msg)

        if not any(isinstance(publisher, ScoreTracker) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to ScoreTracker."
            raise RuntimeError(msg)

        if not any(isinstance(publisher, SpawnStrategyImpl) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to SpawnStrategyImpl."
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        match message:
            case NumClearedLinesMessage(num_cleared_lines=num_cleared_lines):
                self._ui_elements.num_cleared_lines = num_cleared_lines
            case NewLevelMessage(level=level):
                self._ui_elements.level = level
            case ScoreMessage(score=score, session_high_score=session_high_score):
                self._ui_elements.score = score
                self._ui_elements.session_high_score = session_high_score
            case SpawnMessage(next_block=next_block):
                self._ui_elements.next_block = next_block
            case FinishedLineClearMessage(cleared_lines=cleared_lines):
                max_cleared_lines = 4
                if len(cleared_lines) == max_cleared_lines:
                    assert cleared_lines == list(range(cleared_lines[0], cleared_lines[0] + max_cleared_lines))
                    self._ui_elements.animations.append(
                        TetrisAnimationSpec(total_frames=30, top_line_idx=cleared_lines[0])
                    )
            case StartingLineClearMessage():
                pass
            case PowerupTTLsMessage(powerup_ttls=powerup_ttls):
                self._ui_elements.powerup_ttls = powerup_ttls
            case _:
                msg = f"Unexpected message: {message}"
                raise ValueError(msg)

    def update(self, board: NDArray[np.uint8]) -> None:
        self._ui_elements.board = board
        for animation in self._ui_elements.animations:
            animation.advance_frame()

        self._ui_elements.animations = [animation for animation in self._ui_elements.animations if not animation.done]
