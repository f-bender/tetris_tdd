from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.rule import Publisher
from tetris.rules.core.spawn_drop_merge.drop import DropStrategy, DropStrategyImpl
from tetris.rules.core.spawn_drop_merge.merge import MergeStrategy, MergeStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategy, SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategy


class MergeMessage(NamedTuple):
    quick: bool


class SpawnDropMergeRule(Callback, Publisher):
    def __init__(
        self,
        *,
        spawn_delay: int = 25,
        speed_strategy: SpeedStrategy | None = None,
        spawn_strategy: SpawnStrategy | None = None,
        drop_strategy: DropStrategy | None = None,
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Initialize the DropRule.

        Args:
            spawn_delay: Number of frames after a block is merged, before the next block is spawned.
            speed_strategy: Strategy for determining the speed of the block drop/merge.
            spawn_strategy: Strategy for spawning a new block.
            drop_strategy: Strategy for dropping the active block.
            merge_strategy: Strategy for merging the active block into the board.
        """
        super().__init__()

        self._spawn_delay = spawn_delay

        self._spawn_strategy = spawn_strategy or SpawnStrategyImpl()
        self._drop_strategy = drop_strategy or DropStrategyImpl()
        self._merge_strategy = merge_strategy or MergeStrategyImpl()
        self._speed_strategy = speed_strategy or SpeedStrategy()

        self._last_merge_frame: int | None = None
        self._last_drop_frame: int = 0

    def on_game_start(self) -> None:
        self._last_merge_frame = None
        self._last_drop_frame = 0

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        """Do only one of the actions at a time: Either spawn a block, or drop the block, or merge it into the board.

        Block spawning happens `spawn_delay` frames after the last merge, regardless of the quick-drop-action.
        Block dropping and merging acts on the quick interval schedule in case the quick-drop-action is held.
        """
        if not board.has_active_block():
            if not self._last_merge_frame or frame_counter - self._last_merge_frame >= self._spawn_delay:
                self._spawn_strategy.apply(board)
                self._last_drop_frame = frame_counter
            return

        quick_drop_held = action_counter.held_since(Action(down=True)) != 0

        if self._speed_strategy.should_trigger(
            frames_since_last_drop=frame_counter - self._last_drop_frame,
            quick_drop_held=quick_drop_held,
        ):
            merge = self._drop_or_merge(board)

            if merge:
                self._last_merge_frame = frame_counter
                self.notify_subscribers(MergeMessage(quick=quick_drop_held))
            else:  # drop
                self._last_drop_frame = frame_counter

    def _drop_or_merge(self, board: Board) -> bool:
        """Returns bool whether a merge has happened."""
        try:
            self._drop_strategy.apply(board)
        except CannotDropBlockError:
            self._merge_strategy.apply(board)
            return True
        else:
            return False
