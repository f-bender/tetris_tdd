from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.board_manipulation import GradualBoardManipulation
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.core.spawn_drop_merge.drop import DropStrategy, DropStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.merge import MergeStrategy, MergeStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategy, SpawnStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.speed import LineClearSpeedUp, SpeedStrategy, SpeedStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.synchronized_spawn import SynchronizedSpawning
from tetris.game_logic.rules.messages import (
    FinishedMergeMessage,
    Speed,
    StartMergeMessage,
    SynchronizedSpawnCommandMessage,
)


class SpawnDropMergeRule(Callback, Publisher, Subscriber):
    INSTANT_DROP_AND_MERGE_ACTION = Action(down=True, confirm=True)
    QUICK_DROP_ACTION = Action(down=True)

    def __init__(  # noqa: PLR0913
        self,
        *,
        merge_delay: int = 25,
        synchronized_spawn: bool = False,
        spawn_strategy: SpawnStrategy | None = None,
        drop_strategy: DropStrategy | None = None,
        merge_strategy: MergeStrategy | None = None,
        speed_strategy: SpeedStrategy | None = None,
        board_manipulation_after_merge: GradualBoardManipulation | None = None,
        animate_board_manipulation: bool = True,
    ) -> None:
        """Initialize the DropRule.

        Args:
            merge_delay: Number of frames after a block is merged, before the next block may be spawned.
            synchronized_spawn: Whether to synchronize the spawning of new blocks across all games. Requires a
                SynchronizedSpawnRule to be present in the game.
            speed_strategy: Strategy for determining the speed of the block drop/merge.
            spawn_strategy: Strategy for spawning a new block.
            drop_strategy: Strategy for dropping the active block.
            merge_strategy: Strategy for merging the active block into the board.
            board_manipulation_after_merge: Board manipulation to apply after each merge. Defaults to clearing full
                lines.
            animate_board_manipulation: Whether to animate the board manipulation after the merge. If False, the board
                manipulation will be applied instantly.
        """
        super().__init__()

        self._merge_delay = merge_delay
        self._synchronized_spawn = synchronized_spawn

        self.spawn_strategy = spawn_strategy or SpawnStrategyImpl()
        self.drop_strategy = drop_strategy or DropStrategyImpl()
        self.merge_strategy = merge_strategy or MergeStrategyImpl()
        self.speed_strategy = speed_strategy or LineClearSpeedUp(SpeedStrategyImpl())
        self.board_manipulation_after_merge = board_manipulation_after_merge or ClearFullLines()
        self._animate_board_manipulation = animate_board_manipulation

        self._last_merge_start_frame: int | None = None
        self._last_drop_frame: int = 0

        self._received_spawn_command = False

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return (
            self._synchronized_spawn
            and isinstance(publisher, SynchronizedSpawning)
            and publisher.game_index == self.game_index
        )

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if (self._synchronized_spawn and len(publishers) != 1) or (
            not self._synchronized_spawn and len(publishers) != 0
        ):
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if self._synchronized_spawn and isinstance(message, SynchronizedSpawnCommandMessage):
            self._received_spawn_command = True

    def on_game_start(self) -> None:
        self._received_spawn_command = False
        self._last_merge_start_frame = None
        self._last_drop_frame = 0

    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        """Do only one of the actions at a time: Either spawn a block, or drop the block, or merge it into the board.

        (Exception: INSTANT_DROP_AND_MERGE_ACTION allows instantly dropping *and* merging)

        Block spawning happens `spawn_delay` frames after the last merge, regardless of the quick-drop-action.
        Block dropping and merging acts on the quick interval schedule in case the quick-drop-action is held.
        """
        if not board.has_active_block():
            if (
                self._last_merge_start_frame is not None
                and self._animate_board_manipulation
                and (current_frame := frame_counter - self._last_merge_start_frame) <= self._merge_delay
            ):
                self.board_manipulation_after_merge.manipulate_gradually(
                    board,
                    current_frame=current_frame,
                    # "+ 1" because of "fencepost" situation:
                    # the first frame is the merge frame, the last frame is the spawn frame,
                    # merge_delay is the number of frames between them
                    total_frames=self._merge_delay + 1,
                )

            if self._last_merge_start_frame is None:
                # game is just starting
                self.spawn_strategy.apply(board)
                self._last_drop_frame = frame_counter
            elif (frames_since_last_merge_start := frame_counter - self._last_merge_start_frame) >= self._merge_delay:
                if frames_since_last_merge_start == self._merge_delay:
                    self.notify_subscribers(FinishedMergeMessage())
                self._finalize_merge(board, frame_counter)

            return

        quick_drop_held = action_counter.held_since(self.QUICK_DROP_ACTION) != 0

        if not (
            instant_drop_and_merge := action_counter.held_since(self.INSTANT_DROP_AND_MERGE_ACTION) != 0
        ) and not self.speed_strategy.should_trigger(
            frames_since_last_drop=frame_counter - self._last_drop_frame,
            quick_drop_held=quick_drop_held,
        ):
            return

        self._last_drop_frame = frame_counter

        if instant_drop_and_merge:
            self._instant_drop_and_merge(board)
            merge = True
        else:
            merge = self._drop_or_merge(board)

        if not merge:  # i.e. drop
            return

        self._last_merge_start_frame = frame_counter
        self.notify_subscribers(
            StartMergeMessage(
                speed=Speed.INSTANT if instant_drop_and_merge else (Speed.QUICK if quick_drop_held else Speed.NORMAL),
            )
        )
        self.board_manipulation_after_merge.manipulate_gradually(
            board,
            current_frame=0,
            # "+ 1" because of "fencepost" situation:
            # the first frame is the merge frame, the last frame is the spawn frame,
            # merge_delay is the number of frames between them
            total_frames=self._merge_delay + 1 if self._animate_board_manipulation else 1,
        )
        if self._merge_delay == 0:
            self._finalize_merge(board, frame_counter)

    def _finalize_merge(self, board: Board, frame_counter: int) -> None:
        if not self._synchronized_spawn or self._received_spawn_command:
            self.spawn_strategy.apply(board)
            self._received_spawn_command = False
            self._last_drop_frame = frame_counter

    def _drop_or_merge(self, board: Board) -> bool:
        """Returns bool whether a merge has happened."""
        try:
            self.drop_strategy.apply(board)
        except CannotDropBlockError:
            self.merge_strategy.apply(board)
            return True
        else:
            return False

    def _instant_drop_and_merge(self, board: Board) -> None:
        while True:
            try:
                self.drop_strategy.apply(board)
            except CannotDropBlockError:
                self.merge_strategy.apply(board)
                break
