import queue
from typing import NamedTuple, override

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
from tetris.game_logic.rules.messages import MergeMessage, PostMergeFinishedMessage


class PostMergeRule(Publisher, Subscriber, Rule):
    def __init__(self, effect_duration_frames: int = 30) -> None:
        super().__init__()

        self._effect_duration_frames = effect_duration_frames
        self._effect_start_frame: int | None = None

        self._effect_queue = queue.Queue[GradualBoardManipulation]()
        self._current_effect: GradualBoardManipulation | None = None
        self._line_clear_effect = ClearFullLines()

    @override
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        if self._current_effect is None:
            return

        self._effect_start_frame = self._effect_start_frame or frame_counter

        effect_frame = frame_counter - self._effect_start_frame
        self._current_effect.manipulate_gradually(
            board, current_frame=effect_frame, total_frames=self._effect_duration_frames
        )

        if effect_frame >= self._effect_duration_frames - 1:
            self._effect_start_frame = None

            if self._effect_queue.empty():
                self._current_effect = None
                self.notify_subscribers(PostMergeFinishedMessage())
            else:
                self._current_effect = self._effect_queue.get()

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, DropMergeRule) and publisher.game_index == self.game_index

    @override
    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, MergeMessage):
            self._current_effect = self._line_clear_effect

        # TODO: watch for powerup rule message which says to apply e.g. gravity
        # (and in that case, enqueue gravity + line clear)

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if not any(isinstance(p, DropMergeRule) for p in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} is not subscribed to a DropMergeRule: {publishers}"
            raise RuntimeError(msg)
