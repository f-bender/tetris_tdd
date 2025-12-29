from typing import NamedTuple, override

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.interfaces.animations import (
    BlooperAnimationSpec,
    PowerupTriggeredAnimationSpec,
    TetrisAnimationSpec,
)
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.game_logic.rules.messages import (
    BlooperOverlayTrigger,
    ControllerSymbolUpdatedMessage,
    FinishedLineFillMessage,
    NewLevelMessage,
    NumClearedLinesMessage,
    PowerupTriggeredMessage,
    PowerupTTLsMessage,
    ScoreMessage,
    SpawnMessage,
)
from tetris.game_logic.rules.special.powerup_effect import BlooperEffect


class UiAggregator(Subscriber, Callback):
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
        from tetris.controllers.bot_assisted import BotAssistedController
        from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines
        from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
        from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
        from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
        from tetris.game_logic.rules.core.spawn.spawn import SpawnRule
        from tetris.game_logic.rules.special.powerup import PowerupRule

        return publisher.game_index == self.game_index and (
            isinstance(
                publisher,
                ClearedLinesTracker
                | SpawnRule
                | ScoreTracker
                | LevelTracker
                | PowerupRule
                | BotAssistedController
                | BlooperEffect,
            )
            or (isinstance(publisher, FillLines) and publisher.is_line_clearer)
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
        from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
        from tetris.game_logic.rules.core.spawn.spawn import SpawnRule

        if not any(isinstance(publisher, ClearedLinesTracker) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to ClearedLinesTracker."
            raise RuntimeError(msg)

        if not any(isinstance(publisher, ScoreTracker) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to ScoreTracker."
            raise RuntimeError(msg)

        if not any(isinstance(publisher, SpawnRule) for publisher in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} has no subscription to SpawnRule."
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:  # noqa: C901
        match message:
            case NumClearedLinesMessage(num_cleared_lines=num_cleared_lines):
                self._ui_elements.num_cleared_lines = num_cleared_lines
            case NewLevelMessage(level=level):
                self._ui_elements.level = level
            case ScoreMessage(score=score, session_high_score=session_high_score):
                self._ui_elements.score = score
                self._ui_elements.session_high_score = session_high_score
                self._ui_elements.rank = message.rank
            case SpawnMessage(next_block=next_block):
                self._ui_elements.next_block = next_block
            case FinishedLineFillMessage(filled_lines=cleared_lines):
                max_cleared_lines = 4
                if cleared_lines == list(range(cleared_lines[0], cleared_lines[0] + max_cleared_lines)):
                    self._ui_elements.animations.append(
                        TetrisAnimationSpec(total_frames=30, top_line_idx=cleared_lines[0])
                    )
            case PowerupTTLsMessage(powerup_ttls=powerup_ttls):
                self._ui_elements.powerup_ttls = powerup_ttls
            case PowerupTriggeredMessage(position=position):
                self._ui_elements.animations.append(PowerupTriggeredAnimationSpec(total_frames=20, position=position))
            case ControllerSymbolUpdatedMessage(controller_symbol=controller_symbol):
                self._ui_elements.controller_symbol = controller_symbol
            case BlooperOverlayTrigger():
                # ignore it if we are already game over
                if not self._ui_elements.game_over:
                    # make it last for 5 seconds (at 60 fps)
                    self._ui_elements.animations.append(BlooperAnimationSpec(total_frames=300))
            case _:
                pass

    @override
    def on_game_start(self, game_index: int) -> None:
        self._ui_elements.game_over = False

    @override
    def on_game_over(self, game_index: int) -> None:
        self._ui_elements.game_over = True

    def update(self, board: NDArray[np.uint8]) -> None:
        self._ui_elements.board = board
        for animation in self._ui_elements.animations:
            animation.advance_frame()

        self._ui_elements.animations = [animation for animation in self._ui_elements.animations if not animation.done]
