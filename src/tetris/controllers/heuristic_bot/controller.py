"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

import logging
import time
from copy import deepcopy
from threading import Lock, Thread
from typing import Final, NamedTuple

import numpy as np

from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.game_logic.components.block import Block, Vec
from tetris.game_logic.components.board import Board, PositionedBlock
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.rules.core.spawn_drop_merge.spawn import SpawnMessage, SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule

LOGGER = logging.getLogger(__name__)


class Plan(NamedTuple):
    expected_board_before: Board
    target_positioned_block: PositionedBlock


class Misalignment(NamedTuple):
    x: int
    rotation: bool

    def __bool__(self) -> bool:
        return self.x != 0 or self.rotation


class HeuristicBotController(Controller, Subscriber):
    # we assume that the block will spawn in the top _SPAWN_ROWS rows, meaning that we are guaranteed to be able to
    # execute a plan immediately after spawning if these rows are empty
    _SPAWN_ROWS = 2

    def __init__(
        self,
        board: Board,
        heuristic: Heuristic | None = None,
        *,
        lightning_mode: bool = False,
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
            fps: Frames per second that the game will run with. Used as the polling rate of the planning thread. (I.e.
                how frequently to check whether it's time to make the next plan.)
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

        self._lightning_mode = lightning_mode
        self._fps = fps

        if not self._lightning_mode:
            Thread(target=self._continuously_plan, daemon=True).start()

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, SpawnStrategyImpl) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if len(publishers) != 1:
            msg = f"{type(self).__name__} of game {self.game_index} has {len(publishers)} subscriptions: {publishers}"
            raise RuntimeError(msg)

    def notify(self, message: NamedTuple) -> None:
        if not isinstance(message, SpawnMessage):
            return

        # TODO: probably compute plan here in case of lightning mode, de-cluttering get_action
        # directly manipulate board here already, then in get_action just ALWAYS return INSTANT_DROP_AND_MERGE_ACTION?

        with self._planning_lock:
            self._current_block = message.block
            self._next_block = message.next_block

            if not self._current_plan:
                assert not self._next_plan
                # we have crossed over to a new block, but still don't have a plan for the previous one; tell the
                # planning thread to cancel the planning for the previous block (and start planning for the new one)
                self._cancel_planning_flag = True
                if not self._lightning_mode:
                    LOGGER.warning("Previous block was merged before plan for it could even be created")
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
                if not self._lightning_mode:
                    LOGGER.warning("Didn't finish planning for this block before it was spawned")

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
            LOGGER.warning("Need to completely replan as the board is not as expected for the plan")

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

    def get_action(self, board: Board | None = None) -> Action:
        if self._lightning_mode and self._current_block is not None and self._current_plan is None:
            self._current_plan = self._create_plan(self._real_board, self._current_block)

        if (
            self._lightning_mode
            and self._current_plan
            and not np.any(self._real_board.array_view_without_active_block()[: self._SPAWN_ROWS])
            and self._real_board.has_active_block()
        ):
            # "cheat" by instantly placing the block at the target position instead of performing the actions to get it
            # there
            # do this only if the top 2 rows are empty, to avoid cheating ourselves out of a non-navigable situation
            self._real_board.active_block = self._current_plan.target_positioned_block
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
