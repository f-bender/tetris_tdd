from typing import NamedTuple

from tetris.board_manipulations.board_manipulation import BoardManipulation
from tetris.board_manipulations.gravity import Gravity
from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import MergeMessage, SpawnDropMergeRule, Speed


class ParryRule(Subscriber):
    PARRY_ACTION = Action(confirm=True)

    def __init__(self, leeway_frames: int = 1, reward_board_manipulation: BoardManipulation | None = None) -> None:
        super().__init__()
        self._just_merged = False
        self._last_merge_frame: int | None = 0
        self._leeway_frames = leeway_frames
        self._reward_board_manipulation = reward_board_manipulation or Gravity(per_col_probability=0.2)
        self._already_applied = False

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, SpawnDropMergeRule) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        if self._already_applied:
            return

        if self._just_merged:
            self._last_merge_frame = frame_counter
            self._just_merged = False

        if self._last_merge_within_leeway_frames(frame_counter) and self._parry_press_started_within_leeway_frames(
            action_counter,
        ):
            self._reward_board_manipulation.manipulate(board)
            self._already_applied = True

    def _last_merge_within_leeway_frames(self, frame_counter: int) -> bool:
        return self._last_merge_frame is not None and frame_counter - self._last_merge_frame <= self._leeway_frames

    def _parry_press_started_within_leeway_frames(self, action_counter: ActionCounter) -> bool:
        return 0 < action_counter.held_since(self.PARRY_ACTION) <= self._leeway_frames + 1

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, MergeMessage) and message.speed is Speed.NORMAL:
            self._just_merged = True
            self._already_applied = False
