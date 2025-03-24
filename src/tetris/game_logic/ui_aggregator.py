from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.interfaces.animations import LineClearAnimationSpec, TetrisAnimationSpec
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.rules.core.messages import LineClearMessage, MergeMessage, SpawnMessage
from tetris.rules.monitoring.track_score_rule import ScoreMessage


class UiAggregator(Subscriber):
    """Subscriber to all UI-relevant events, aggregating them into UiElements."""

    def __init__(self, board: NDArray[np.uint8]) -> None:
        super().__init__()
        self._ui_elements = SingleUiElements(board=board)
        self._frame_counter = 0
        self._next_spawn_frame_counter: int | None = None

    def reset(self) -> None:
        self._ui_elements.reset()

    @property
    def ui_elements(self) -> SingleUiElements:
        return self._ui_elements

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
        from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
        from tetris.rules.monitoring.track_score_rule import TrackScoreRule

        return (
            isinstance(publisher, TrackScoreRule | SpawnStrategyImpl | ClearFullLinesRule)
            and publisher.game_index == self.game_index
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 3:  # noqa: PLR2004
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}. "
                "Expected 3 subscriptions (TrackScoreRule, SpawnStrategyImpl, ClearFullLinesRule)."
            )
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        match message:
            case ScoreMessage(score=score):
                self._ui_elements.score = score
            case SpawnMessage(next_block=next_block):
                self._ui_elements.next_block = next_block
            case MergeMessage(next_spawn_frame_counter=next_spawn_frame_counter):
                self._next_spawn_frame_counter = next_spawn_frame_counter
            # TODO: merge line clearing logic into the SpawnDropMergeRule; they are just too intertwined!
            # actually though, think about it. Can I keep it flexible enough that I can change the rules, e.g. making it
            # such that a full line isn't immediately cleared, but you have to fill a chunk of 4 lines for them to be
            # cleared? With this in mind, think again if this can be achieved when moving line clear rule into spawn
            # drop merge rule, or if I should keep it separate (note: line clear message should actually correspond to
            # lines being cleared, not to lines being filled (it's just that with the standard rules, those are the same))
            case LineClearMessage(cleared_lines=cleared_lines):
                if self._next_spawn_frame_counter is not None and self._next_spawn_frame_counter > self._frame_counter:
                    self._ui_elements.animations.append(
                        LineClearAnimationSpec(
                            total_frames=self._next_spawn_frame_counter - self._frame_counter,
                            cleared_lines=cleared_lines,
                        )
                    )

                max_cleared_lines = 4
                if len(cleared_lines) == max_cleared_lines:
                    assert cleared_lines == list(range(cleared_lines[0], cleared_lines[0] + max_cleared_lines))
                    self._ui_elements.animations.append(
                        TetrisAnimationSpec(total_frames=30, top_line_idx=cleared_lines[0])
                    )
            case _:
                msg = f"Unexpected message: {message}"
                raise ValueError(msg)

    def update(self, board: NDArray[np.uint8], frame_counter: int) -> None:
        self._frame_counter = frame_counter
        self._ui_elements.board = board
        for animation in self._ui_elements.animations:
            animation.advance_frame()

        self._ui_elements.animations = [animation for animation in self._ui_elements.animations if not animation.done]
