from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.core.drop_merge.drop import DropStrategy, DropStrategyImpl
from tetris.game_logic.rules.core.drop_merge.merge import MergeStrategy, MergeStrategyImpl
from tetris.game_logic.rules.core.drop_merge.speed import LevelSpeedUp, SpeedStrategy
from tetris.game_logic.rules.messages import MergeMessage, Speed


class DropMergeRule(Callback, Publisher):
    INSTANT_DROP_AND_MERGE_ACTION = Action(down=True, confirm=True)
    QUICK_DROP_ACTION = Action(down=True)

    def __init__(
        self,
        *,
        drop_strategy: DropStrategy | None = None,
        merge_strategy: MergeStrategy | None = None,
        speed_strategy: SpeedStrategy | None = None,
    ) -> None:
        """Initialize the DropMergeRule.

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

        self.drop_strategy = drop_strategy or DropStrategyImpl()
        self.merge_strategy = merge_strategy or MergeStrategyImpl()
        self.speed_strategy = speed_strategy or LevelSpeedUp()

        self._last_drop_frame: int | None = None

    def on_game_start(self) -> None:
        self._last_drop_frame = None

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
            return

        if self._last_drop_frame is None:
            # first call after spawn - treat as if last drop was just now
            self._last_drop_frame = frame_counter

        instant_drop_and_merge = action_counter.held_since(self.INSTANT_DROP_AND_MERGE_ACTION) != 0
        quick_drop_held = action_counter.held_since(self.QUICK_DROP_ACTION) != 0

        if not instant_drop_and_merge and not self.speed_strategy.should_trigger(
            frames_since_last_drop=frame_counter - self._last_drop_frame, quick_drop_held=quick_drop_held
        ):
            return

        if instant_drop_and_merge:
            self._instant_drop_and_merge(board)
            merge = True
        else:
            merge = self._drop_or_merge(board)

        if merge:
            self._last_drop_frame = None
            self.notify_subscribers(
                MergeMessage(
                    speed=Speed.INSTANT
                    if instant_drop_and_merge
                    else (Speed.QUICK if quick_drop_held else Speed.NORMAL)
                )
            )
        else:
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
