"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

import logging
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock, Thread
from typing import Final, NamedTuple, cast

import numpy as np

from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.game_logic.components.block import Block, Vec
from tetris.game_logic.components.board import Board, PositionedBlock
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.game_logic.rules.messages import SpawnMessage

LOGGER = logging.getLogger(__name__)


class Plan(NamedTuple):
    expected_board_before: Board
    target_positioned_block: PositionedBlock


class Misalignment(NamedTuple):
    x: int
    rotation: bool

    def __bool__(self) -> bool:
        return self.x != 0 or self.rotation


# TODO refactor; split into (at least) 2 classes
class HeuristicBotController(Controller, Subscriber):
    # we assume that the block will spawn in the top _SPAWN_ROWS rows, meaning that we are guaranteed to be able to
    # execute a plan immediately after spawning if these rows are empty
    _SPAWN_ROWS = 2

    def __init__(  # noqa: PLR0913
        self,
        board: Board,
        heuristic: Heuristic | None = None,
        *,
        lightning_mode: bool = False,
        process_pool: ProcessPoolExecutor | None = None,
        ensure_consistent_behaviour: bool = False,
        fps: float = 60,
    ) -> None:
        """Initialize the bot controller.

        Args:
            board: The board that is actually being played on.
            heuristic: The heuristic to use for evaluating the board state. If None, the default heuristic is used.
            lightning_mode: For fastest possible simulation of bot gameplay, without regard for frame time constraints.
                When active, the bot will eagerly plan where to put a block immediately after it has spawned, with
                the get_action method. It will then instantly place the block at the target position instead of
                performing the actions to get it there, basically "cheating" the controller system by directly
                manipulating the active block on the board.
                Otherwise, when inactive, a thread is started which computes plans concurrently without blocking
                gameplay, and get_action executes a plan by selecting appropriate actions one by one, and is guaranteed
                to return very quickly (the default).
            process_pool: Optional process pool to hand work packets during planning to. Allows for some parallelization
                of the planning of even a single HeuristicBotController, empirically leading to a ~2.2x speedup.
                When multiple HeuristicBotControllers are run concurrently, the same process pool should be handed to
                all of them such that the one pool can handle the scheduling.
            ensure_consistent_behaviour: Whether to ensure that given the same board and active block, the bot will
                consistently choose the same location to place the block at. Note: only has an effect when
                `process_pool` is provided, otherwise behaviour is consistent anyways.
            fps: Frames per second that the game will run with. Used as the polling rate of the planning thread. (I.e.
                how frequently to check whether it's time to make the next plan.) Ignored in lightning mode.

        Note that it is inefficient to use both lightning mode and process pool with multiple
        HeuristicBotController-controlled games in the same thread since lightning mode avoids creating an internal
        thread, and thus preventing multiple HeuristicBotControllers from dispatching tasks to the process pool at the
        same time, making it underutilized.
        Lightning mode plus process pool does still make sense for multiple HeuristicBotControllers, if their games are
        being run in separate threads.
        """
        super().__init__()
        self._real_board: Final = board
        self.heuristic = heuristic or Heuristic()  # type: ignore[call-arg]

        self._current_block: Block | None = None
        self._next_block: Block | None = None

        self._planning_lock = Lock()
        self._current_plan: Plan | None = None
        self._next_plan: Plan | None = None
        self._cancel_planning_flag: bool = False

        self._active_frame: bool = True

        self._fps = fps

        self._lightning_mode = lightning_mode
        self._ready_for_instand_drop_and_merge = False

        self._process_pool = process_pool
        self._ensure_consistent_behaviour = ensure_consistent_behaviour

        if not self._lightning_mode:
            Thread(target=self._continuously_plan, daemon=True).start()

    @property
    def lightning_mode(self) -> bool:
        return self._lightning_mode

    @property
    def is_using_process_pool(self) -> bool:
        return self._process_pool is not None

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, SpawnStrategyImpl) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if not isinstance(message, SpawnMessage):
            return

        if self._lightning_mode:
            self._handle_block_spawn_lightning(message.block)
        else:
            self._handle_block_spawn_non_lightning(current_block=message.block, next_block=message.next_block)

    def _handle_block_spawn_lightning(self, current_block: Block) -> None:
        self._current_plan = self._create_plan(self._real_board, current_block)

        if self._current_plan and not np.any(self._real_board.array_view_without_active_block()[: self._SPAWN_ROWS]):
            # "cheat" by instantly placing the block at the target position instead of performing the actions to get
            # it there
            # do this only if the top 2 rows are empty, to avoid cheating our way out of a non-navigable situation
            self._real_board.active_block = self._current_plan.target_positioned_block
            self._ready_for_instand_drop_and_merge = True

    def _handle_block_spawn_non_lightning(self, current_block: Block, next_block: Block) -> None:
        with self._planning_lock:
            self._current_block = current_block
            self._next_block = next_block

            if not self._current_plan:
                assert not self._next_plan
                # we have crossed over to a new block, but still don't have a plan for the previous one; tell the
                # planning thread to cancel the planning for the previous block (and start planning for the new one)
                self._cancel_planning_flag = True
                LOGGER.debug("Previous block was merged before plan for it could even be created")
            elif self._next_plan:
                assert self._current_plan
                # planning has finished in time; we know that we are not currently in the process of creating a plan

                self._set_current_plan(self._next_plan)

                # let the planning thread know to start planning again by setting _next_plan to None
                self._next_plan = None
            else:
                # planning ahead has not finished in time; we are currently in the process of creating the current plan
                # let the planning thread know to directly set _current_plan (by having it be None)
                self._current_plan = None
                LOGGER.debug("Didn't finish planning for this block before it was spawned")

    def _set_current_plan(self, plan: Plan) -> None:
        """Set the current plan to the given plan, triggering the execution of the plan in get_action.

        It is checked whether the expected board state before the plan matches the current real board state.
        If this is not the case, the current and next plans are both set to None, and the bot replans from scratch.
        """
        if np.array_equal(
            plan.expected_board_before.array_view_without_active_block(),
            self._real_board.array_view_without_active_block(),
        ):
            # start executing the plan (in get_action) by setting _current_plan to the next plan
            self._current_plan = plan
        else:
            # we have made a false assumption about the board state; don't execute any plan, and replan from
            # scratch
            self._current_plan = self._next_plan = None
            LOGGER.debug("Need to completely replan as the board is not as expected for the plan")

    def _continuously_plan(self) -> None:
        while self._current_block is None or self._next_block is None:
            # wait until the first block has spawned
            time.sleep(1 / self._fps)

        expected_board_after_latest_plan = Board()
        while True:
            while self._next_plan is not None:
                assert self._current_plan is not None
                # all the planning has been done in time; wait for the next block to spawn, that will need to be planned
                # for again
                time.sleep(1 / self._fps)

            with self._planning_lock:
                if self._current_plan is None:
                    # plan for the current block
                    block = self._current_block
                    expected_board_before = deepcopy(self._real_board)
                else:
                    # plan for the current block exists already: plan ahead for the next block
                    block = self._next_block
                    expected_board_before = deepcopy(expected_board_after_latest_plan)

            plan = self._create_plan(expected_board_before, block, expected_board_after_latest_plan)
            if plan is None:
                # planning was cancelled, or no valid plan was found
                continue

            with self._planning_lock:
                if self._current_plan is None:
                    # EITHER
                    # we were planning for the current block from the start
                    # OR
                    # while we were planning ahead for the next block, the current block was merged, so the block we
                    # have planned for is now the current block
                    self._set_current_plan(plan)
                else:
                    self._next_plan = plan

    def _create_plan(
        self, expected_board_before: Board, block: Block, expected_board_after: Board | None = None
    ) -> Plan | None:
        """Create a plan to fit a block into the optimal position given the board state.

        Args:
            expected_board_before: The board state based on which the plan should be created.
            block: The block that a plan should be created for.
            expected_board_after: Output argument that will be set to the expected board state after the plan has been
                executed, if provided.

        Returns:
            Plan: The plan to fit the block into the optimal position, consisting of the expected board state before the
                plan is to be executed (for double checking), and the target positioned block (i.e. target position and
                rotation of the block).
        """
        if self._process_pool is None:
            return self._create_plan_single_process(expected_board_before, block, expected_board_after)
        return self._create_plan_with_process_pool(expected_board_before, block, expected_board_after)

    def _create_plan_single_process(
        self, expected_board_before: Board, block: Block, expected_board_after: Board | None = None
    ) -> Plan | None:
        internal_planning_board = Board()
        min_loss: float | None = None
        min_loss_positioned_block: PositionedBlock | None = None

        for rotated_block in block.unique_rotations():
            top_offset, left_offset, _, right_offset = rotated_block.actual_bounding_box

            for x_position in range(-left_offset, expected_board_before.width - right_offset + 1):
                if self._cancel_planning_flag:
                    self._cancel_planning_flag = False
                    return None

                positioned_block = PositionedBlock(rotated_block, Vec(y=-top_offset, x=x_position))

                internal_planning_board.set_from_other(expected_board_before)
                if internal_planning_board.positioned_block_overlaps_with_active_cells(positioned_block):
                    continue

                internal_planning_board.active_block = positioned_block

                while True:
                    try:
                        internal_planning_board.drop_active_block()
                    except CannotDropBlockError:
                        internal_planning_board.merge_active_block()
                        internal_planning_board.clear_lines(internal_planning_board.get_full_line_idxs())
                        break

                loss = self.heuristic.loss(internal_planning_board)
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_loss_positioned_block = positioned_block
                    if expected_board_after is not None:
                        expected_board_after.set_from_other(internal_planning_board)

        if min_loss_positioned_block is None:
            return None

        return Plan(expected_board_before, min_loss_positioned_block)

    def _create_plan_with_process_pool(  # noqa: C901
        self, expected_board_before: Board, block: Block, expected_board_after: Board | None = None
    ) -> Plan | None:
        assert self._process_pool is not None

        # if process pool is already fairly saturated with tasks, give it one large task to reduce overhead
        # NOTE: checking for half of processes being occupied (-> `// 2`) performed well empirically
        if len(self._process_pool._pending_work_items) >= self._process_pool._max_workers // 2:  # type: ignore[attr-defined]  # noqa: SLF001
            plan_result = self._process_pool.submit(
                self._plan_from_board_and_block,
                expected_board_before,
                block,
                self.heuristic,
                return_board_after=expected_board_after is not None,
            ).result()
            if plan_result is None:
                return None

            if expected_board_after is None:
                return cast("Plan", plan_result)

            plan, board_after = cast("tuple[Plan, Board]", plan_result)
            expected_board_after.set_from_other(board_after)

            return plan

        loss_futures: list[Future[tuple[Board, PositionedBlock, float] | None]] = []
        for rotated_block in block.unique_rotations():
            top_offset, left_offset, _, right_offset = rotated_block.actual_bounding_box

            loss_futures.extend(
                self._process_pool.submit(
                    self._loss_from_positioned_block,
                    # in moving this to the worker process, a copy is being created anyways, so no need to explicitly
                    # create a copy myself; expected_board_before will stay unchanged in the main process
                    expected_board_before,
                    PositionedBlock(rotated_block, Vec(y=-top_offset, x=x_position)),
                    self.heuristic,
                )
                for x_position in range(-left_offset, expected_board_before.width - right_offset + 1)
            )

        min_loss: float | None = None
        min_loss_positioned_block: PositionedBlock | None = None

        for future in loss_futures if self._ensure_consistent_behaviour else as_completed(loss_futures):
            # NOTE: if we get the futures in an arbitrary order (as completed), two block placements having the
            # same optimal loss will result in one of them being chosen "randomly" (the one that finishes first)
            # -> inconsistent behaviour, even given the same seed
            loss_result = future.result()

            if self._cancel_planning_flag:
                self._cancel_planning_flag = False
                for f in loss_futures:
                    f.cancel()
                return None

            if loss_result is None:
                continue

            board_after, positioned_block, loss = loss_result
            if min_loss is not None and loss >= min_loss:
                continue

            min_loss = loss
            min_loss_positioned_block = positioned_block
            if expected_board_after is not None:
                expected_board_after.set_from_other(board_after)

        if min_loss_positioned_block is None:
            return None

        return Plan(expected_board_before, min_loss_positioned_block)

    @staticmethod
    def _plan_from_board_and_block(
        expected_board_before: Board, block: Block, heuristic: Heuristic, *, return_board_after: bool
    ) -> Plan | tuple[Plan, Board] | None:
        internal_planning_board = Board()
        expected_board_after = Board() if return_board_after else None

        min_loss: float | None = None
        min_loss_positioned_block: PositionedBlock | None = None

        for rotated_block in block.unique_rotations():
            top_offset, left_offset, _, right_offset = rotated_block.actual_bounding_box

            for x_position in range(-left_offset, expected_board_before.width - right_offset + 1):
                positioned_block = PositionedBlock(rotated_block, Vec(y=-top_offset, x=x_position))

                internal_planning_board.set_from_other(expected_board_before)
                if internal_planning_board.positioned_block_overlaps_with_active_cells(positioned_block):
                    continue

                HeuristicBotController._instant_drop_and_merge_block(
                    board=internal_planning_board, positioned_block=positioned_block
                )

                loss = heuristic.loss(internal_planning_board)
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_loss_positioned_block = positioned_block
                    if expected_board_after:
                        expected_board_after.set_from_other(internal_planning_board)

        if min_loss_positioned_block is None:
            return None

        if expected_board_after:
            return Plan(expected_board_before, min_loss_positioned_block), expected_board_after
        return Plan(expected_board_before, min_loss_positioned_block)

    @staticmethod
    def _loss_from_positioned_block(
        board: Board, positioned_block: PositionedBlock, heuristic: Heuristic
    ) -> tuple[Board, PositionedBlock, float] | None:
        if board.positioned_block_overlaps_with_active_cells(positioned_block):
            return None

        HeuristicBotController._instant_drop_and_merge_block(board=board, positioned_block=positioned_block)

        return board, positioned_block, heuristic.loss(board)

    @classmethod
    def _instant_drop_and_merge_block(cls, board: Board, positioned_block: PositionedBlock) -> None:
        """Drop treat `positioned_block` as the active block of `board` and drop it straight down into place.

        When dropped into place, merge the block into place and clear full lines.

        When the rows where blocks can spawn are not empty, the naive and reliable implementation is used.
        Otherwise, an optimized implementation is used which assumes that the rows of the positioned block and above
        are empty.
        """
        if np.any(board.array_view_without_active_block()[: cls._SPAWN_ROWS]):
            cls._instant_drop_and_merge_block_naive(board, positioned_block)
        else:
            cls._instant_drop_and_merge_optimized(board, positioned_block)

    @staticmethod
    def _instant_drop_and_merge_block_naive(board: Board, positioned_block: PositionedBlock) -> None:
        board.active_block = positioned_block
        while True:
            try:
                board.drop_active_block()
            except CannotDropBlockError:
                board.merge_active_block()
                board.clear_lines(board.get_full_line_idxs())
                break

    @staticmethod
    def _instant_drop_and_merge_optimized(board: Board, positioned_block: PositionedBlock) -> None:
        """An optimized version of _instant_drop_and_merge_block_naive.

        This implementation does the same thing in a more computationally efficient way:
        It computes the drop distance and directly places the block at the position it should ultimately drop down to.
        """
        block_actual_cells = positioned_block.block.actual_cells
        block_actual_position = positioned_block.actual_bounding_box.top_left
        positioned_block_lower_bound = (
            block_actual_position.y + block_actual_cells.shape[0] - np.argmax(np.flipud(block_actual_cells), axis=0)
        )

        board_relevant_columns_bool = board.array_view_without_active_block()[
            :,
            block_actual_position.x : block_actual_position.x + block_actual_cells.shape[1],
        ].astype(bool)

        board_upper_bound_under_block = np.where(
            np.any(board_relevant_columns_bool, axis=0),
            np.argmax(
                board_relevant_columns_bool,
                axis=0,
            ),
            board_relevant_columns_bool.shape[0],
        )

        drop_distance = np.min(board_upper_bound_under_block - positioned_block_lower_bound)
        assert drop_distance >= 0

        board.active_block = PositionedBlock(
            block=positioned_block.block,
            position=Vec(positioned_block.position.y + drop_distance, positioned_block.position.x),
        )

        board.merge_active_block()
        board.clear_lines(board.get_full_line_idxs())

    def get_action(self, board: Board | None = None) -> Action:
        if self._ready_for_instand_drop_and_merge:
            # lightning mode has set up the block for us, so we can instantly drop and merge it
            self._ready_for_instand_drop_and_merge = False
            return SpawnDropMergeRule.INSTANT_DROP_AND_MERGE_ACTION

        if self._current_plan is None or self._real_board.active_block is None:
            self._active_frame = True
            return Action()

        # if no misalignment: the complicated part of the current plan has been executed, now simply quick-drop the
        # block into place (ignore self._active_frame; we can quick-drop immediately in a single frame)
        misalignment = self._positioned_blocks_misalignment(
            self._real_board.active_block,
            self._current_plan.target_positioned_block,
        )
        if not misalignment:
            self._active_frame = True
            # only quickdrop if we have already computed the next plan; otherwise we want to buy some time for the next
            # plan to be computed (i.e. slow drop)
            return (
                SpawnDropMergeRule.INSTANT_DROP_AND_MERGE_ACTION
                if self._next_plan is not None or self._lightning_mode
                else Action()
            )

        # skip every other frame such that each action is interpreted as a separate button press, instead of a single
        # button hold
        if not self._active_frame:
            self._active_frame = True
            return Action()

        self._active_frame = False

        move = Action() if misalignment.x == 0 else (Action(left=True) if misalignment.x < 0 else Action(right=True))
        rotate = Action(right_shoulder=True) if misalignment.rotation else Action()

        return move | rotate

    @staticmethod
    def _positioned_blocks_misalignment(block_1: PositionedBlock, block_2: PositionedBlock) -> Misalignment:
        """Returns the misalignment between two blocks.

        Specifically, the x difference in the position (block_2 - block_1),
        and whether the blocks are rotated differently.
        y position is ignored, as it is not relevant for the misalignment.
        """
        return Misalignment(
            x=block_2.position.x - block_1.position.x,
            rotation=not np.array_equal(block_2.block.cells, block_1.block.cells),
        )
