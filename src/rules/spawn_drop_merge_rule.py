from typing import Callable, Protocol

from game_logic.action_counter import ActionCounter
from game_logic.components.block import Block
from game_logic.components.board import Board
from game_logic.components.exceptions import CannotDropBlock, CannotSpawnBlock
from game_logic.game import GameOver
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.controller import Action


class SpawnStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class SpawnStrategyImpl:
    def __init__(self, select_block_fn: Callable[[], Block] = Block.create_random) -> None:
        self._select_block_fn = select_block_fn

    def set_select_block_fn(self, select_block_fn: Callable[[], Block]) -> None:
        self._select_block_fn = select_block_fn

    def apply(self, board: Board) -> None:
        try:
            board.spawn(self._select_block_fn())
        except CannotSpawnBlock:
            raise GameOver


class DropStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class DropStrategyImpl:
    def apply(self, board: Board) -> None:
        board.drop_active_block()


class MergeStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class MergeStrategyImpl:
    def apply(self, board: Board) -> None:
        board.merge_active_block()


class SpawnDropMergeRule:
    def __init__(
        self,
        normal_interval: int = 25,
        quick_interval_factor: float = 8,
        spawn_delay: int | None = None,
        spawn_strategy: SpawnStrategy = SpawnStrategyImpl(),
        drop_strategy: DropStrategy = DropStrategyImpl(),
        merge_strategy: MergeStrategy = MergeStrategyImpl(),
    ) -> None:
        """Initialize the DropRule.

        Args:
            drop_interval_frames: Number of frames between drops while the quick-drop-action is *not* held. The default
                value of 25 is fine-tuned for 60 FPS gameplay.
            quick_drop_factor: `drop_interval_frames` is divided by this factor to obtain the number of frames between
                drops while the quick-drop-action *is* held.
            spawn_delay: Number of frames after a block is merged, before the next block is spawned.
        """
        self._quick_interval_factor = quick_interval_factor
        self.set_interval(normal_interval)

        self._spawn_strategy = spawn_strategy
        self._drop_strategy = drop_strategy
        self._merge_strategy = merge_strategy

        self._spawn_delay = spawn_delay or normal_interval
        self._last_merge_frame: int | None = None

    def set_interval(self, normal_interval: int) -> None:
        self._normal_interval = normal_interval
        self._quick_interval = max(round(normal_interval / self._quick_interval_factor), 1)

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        """Do only one of the actions at a time: Either spawn a block, or drop the block, or merge it into the board.

        Block spawning happens `spawn_delay` frames after the last merge, regardless of the quick-drop-action.
        Block dropping and merging acts on the quick interval schedule in case the quick-drop-action is held.
        """
        if not board.has_active_block():
            if not self._last_merge_frame or frame_counter - self._last_merge_frame >= self._spawn_delay:
                self._spawn_strategy.apply(board)
            return

        if (quick_drop_held_since := action_counter.held_since(Action(down=True))) != 0:
            if self._is_quick_action_frame(quick_drop_held_since):
                if self._drop_or_merge(board):
                    self._last_merge_frame = frame_counter
                    callback_collection.custom_message("quick merge")
        elif self._is_normal_action_frame(frame_counter):
            if self._drop_or_merge(board):
                self._last_merge_frame = frame_counter
                callback_collection.custom_message("slow merge")

    def _is_normal_action_frame(self, frame_counter: int) -> bool:
        return frame_counter % self._normal_interval == 0

    def _is_quick_action_frame(self, quick_drop_held_since: int) -> bool:
        return quick_drop_held_since != 0 and (quick_drop_held_since - 1) % self._quick_interval == 0

    def _drop_or_merge(self, board: Board) -> bool:
        """Returns bool whether a merge has happened."""
        try:
            self._drop_strategy.apply(board)
            return False
        except CannotDropBlock:
            self._merge_strategy.apply(board)
            return True
