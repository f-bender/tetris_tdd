import queue
import random
from functools import partial
from typing import NamedTuple, override

import numpy as np

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.block import BlockType
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines
from tetris.game_logic.rules.board_manipulations.gravity import Gravity
from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
from tetris.game_logic.rules.messages import (
    FillLinesEffectTrigger,
    GravityEffectTrigger,
    MergeMessage,
    PostMergeFinishedMessage,
)
from tetris.game_logic.rules.special.powerup_effect import FillLinesEffect, GravityEffect


class PostMergeRule(Publisher, Subscriber, Callback, Rule):
    def __init__(self, effect_duration_frames: int = 30, minimum_delay_frames: int = 30) -> None:
        if effect_duration_frames <= 0:
            msg = "effect_duration_frames must be positive"
            raise ValueError(msg)

        if minimum_delay_frames <= 0:
            msg = "minimum_delay_frames must be positive"
            raise ValueError(msg)

        super().__init__()

        self._effect_duration_frames = effect_duration_frames
        self._minimum_delay_frames = minimum_delay_frames
        self._effect_start_frame: int | None = None
        self._post_merge_start_frame: int | None = None

        self._effect_queue = queue.Queue[GradualBoardManipulation]()
        self._current_effect: GradualBoardManipulation | None = None

        self._line_clear_effect = FillLines.clear_full_lines()
        self._line_fill_effect = FillLines(
            line_idx_factory=partial(self._get_random_non_empty_lines, num_lines=1),
            fill_value=BlockType.S,
        )
        self._gravity_effect = Gravity()

    @override
    def on_game_start(self) -> None:
        self._effect_queue = queue.Queue[GradualBoardManipulation]()
        self._current_effect = None
        self._effect_start_frame = None
        self._post_merge_start_frame = None

    @override
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        if self._current_effect is None:
            return

        self._post_merge_start_frame = self._post_merge_start_frame or frame_counter
        self._effect_start_frame = self._effect_start_frame or frame_counter

        effect_frame = frame_counter - self._effect_start_frame
        self._current_effect.manipulate_gradually(
            board, current_frame=effect_frame, total_frames=self._effect_duration_frames
        )

        if effect_frame >= self._effect_duration_frames - 1 or (
            self._current_effect.done_already()
            and frame_counter - self._post_merge_start_frame >= self._minimum_delay_frames
        ):
            self._effect_start_frame = None

            if self._effect_queue.empty():
                self._current_effect = self._post_merge_start_frame = None
                self.notify_subscribers(PostMergeFinishedMessage())
            else:
                self._current_effect = self._effect_queue.get()

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return (
            isinstance(publisher, DropMergeRule | GravityEffect | FillLinesEffect)
            and publisher.game_index == self.game_index
        )

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case MergeMessage():
                self._current_effect = self._line_clear_effect
            case GravityEffectTrigger(per_col_probability=per_col_probability):
                self._gravity_effect.per_col_probability = per_col_probability
                self._effect_queue.put(self._gravity_effect)
                self._effect_queue.put(self._line_clear_effect)
            case FillLinesEffectTrigger(num_lines=num_lines):
                self._line_fill_effect.line_idx_factory = partial(self._get_random_non_empty_lines, num_lines=num_lines)
                self._effect_queue.put(self._line_fill_effect)
                self._effect_queue.put(self._line_clear_effect)
            case _:
                msg = f"Unexpected message type {type(message)} received by {type(self).__name__}"
                raise RuntimeError(msg)

    @staticmethod
    def _get_random_non_empty_lines(board: Board, num_lines: int) -> list[int]:
        board_array = board.array_view_without_active_block()
        non_empty_lines = np.where(board_array.any(axis=1))[0].tolist()

        return random.sample(non_empty_lines, k=min(num_lines, len(non_empty_lines)))

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if not any(isinstance(p, DropMergeRule) for p in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} is not subscribed to a DropMergeRule: {publishers}"
            raise RuntimeError(msg)
