"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

import logging
import time
from copy import deepcopy
from threading import Lock, Thread
from typing import Final, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block, Vec
from tetris.game_logic.components.board import Board, PositionedBlock
from tetris.game_logic.components.exceptions import CannotDropBlockError
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.rule import Subscriber
from tetris.rules.core.spawn_drop_merge.spawn import SpawnMessage

LOGGER = logging.getLogger(__name__)


class Plan(NamedTuple):
    expected_board_before: Board
    target_positioned_block: PositionedBlock


class Loss(NamedTuple):
    """Measure of how bad a board is.

    `<` , `>` should be used to compare losses.
    This will use the tuple implementation, meaning that the first element is the most important, and only in case
    of equality are subsequent elements checked.
    """

    # TODO:
    # - write tests for this class!
    # - especially square (O) blocks seem to sometimes be placed at suboptimal spots, where a different spot would have
    #   the same overhangs (none), the same sum of heights, but less adjacent height differences!
    #   -> bug in sum_adjacent_height_differences? Or somewhere else?
    # - punish very high stacks (< 4 cells below top of board)
    # - avoid overhang with larger amount of empty cells below (-> avoid closing a high gap that multiple I-shapes
    #   should go into)
    #   -> probably simply count all empty cells that have an active cells above, AND all active cells that have an
    #      empty cell below, and sum all that
    # - avoid very large adjacent differences; rather have multiple small differences (<=2)
    #   instead of one large one (>=3) (-> prefer gentler slopes)
    # - issue: if there is an overhang, it doesn't even want to put a vertical I there to clear 3 lines (because this
    #   still increase the number of overhaning blocks by 1); this is not ideal
    # - avoid creating new overhangs - rather increase number of blocks on an existing overhang rather than creating
    #   a new one -> count overhangs -> count every overhang separately, even multiple overhangs in the same column!

    num_overhanging_cells: int
    sum_of_cell_heights: int
    sum_of_adjacent_height_differences: int

    @classmethod
    def from_board(cls, board: Board) -> Self:
        board_array = board.as_array_without_active_block().astype(bool)

        return cls(
            cls.count_overhanging_cells(board_array),
            cls.sum_cell_heights(board_array),
            cls.sum_adjacent_height_differences(board_array),
        )

    @staticmethod
    def count_overhanging_cells(board_array: NDArray[np.bool]) -> int:
        # go through each column from bottom to top, find the first "False" (empty cell), and count all "True"s above
        return sum(np.sum(column[np.argmax(~column) :]) for column in np.rot90(board_array, k=-1))

    @staticmethod
    def sum_cell_heights(board_array: NDArray[np.bool]) -> int:
        # get the y indices of all active cells and compute their sum
        # the result is the summed distance of the active cells to the top of the board, so subtract the result from the
        # board height multiplied by the number of active cells to get the summed cell heights counting from the bottom
        y_indices = np.nonzero(board_array)[0]
        return int(y_indices.size * board_array.shape[0] - np.sum(y_indices))

    @staticmethod
    def sum_adjacent_height_differences(board_array: NDArray[np.bool]) -> int:
        # get the highest active cell index for each column, compute adjacent differences, return their absolute sum
        return np.sum(np.abs(np.diff(np.argmax(board_array, axis=0))))


class Misalignment(NamedTuple):
    x: int
    rotation: bool

    def __bool__(self) -> bool:
        return self.x != 0 or self.rotation


class HeuristicBotController(Controller, Subscriber):
    def __init__(self, board: Board, fps: float = 60) -> None:
        """Initializes the bot controller.

        Args:
            board: The board that is actually being played on.
            fps: Frames per second that the game will run with. Used as the polling rate of the planning thread. (I.e.
                how frequently to check whether it's time to make the next plan.)
        """
        self._real_board: Final = board

        self._current_block: Block | None = None
        self._next_block: Block | None = None

        self._planning_lock = Lock()
        self._current_plan: Plan | None = None
        self._next_plan: Plan | None = None
        self._cancel_planning_flag: bool = False

        self._active_frame: bool = True

        self._fps = fps
        Thread(target=self._continuously_plan, daemon=True).start()

    def notify(self, message: NamedTuple) -> None:
        if not isinstance(message, SpawnMessage):
            return

        with self._planning_lock:
            self._current_block = message.block
            self._next_block = message.next_block

            if not self._current_plan:
                assert not self._next_plan
                # we have crossed over to a new block, but still don't have a plan for the previous one; tell the
                # planning thread to cancel the planning for the previous block (and start planning for the new one)
                self._cancel_planning_flag = True
                LOGGER.warning("Previous block was merged before plan for it could even be created")
            elif self._next_plan:
                assert self._current_plan
                # planning has finished in time; we know that we are not currently in the process of creating a plan

                self._set_current_plan(self._next_plan)

                # let the planning thread know to start planning again by setting _next_plan to None
                self._next_plan = None
            else:
                # planning has not finished in time; we are currently in the process of creating the current plan
                # let the planning thread know to directly set _current_plan (by having it be None)
                self._current_plan = None
                LOGGER.warning("Didn't finish planning for this block before it was spawned")

    def _set_current_plan(self, plan: Plan) -> None:
        """Set the current plan to the given plan, triggering the execution of the plan in get_action.

        It is checked whether the expected board state before the plan matches the current real board state.
        If this is not the case, the current and next plans are both set to None, and the bot replans from scratch.
        """
        if np.array_equal(
            plan.expected_board_before.as_array_without_active_block(),
            self._real_board.as_array_without_active_block(),
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
                    block = self._current_block
                    expected_board_before = deepcopy(self._real_board)
                else:
                    block = self._next_block
                    expected_board_before = deepcopy(expected_board_after_latest_plan)

            plan = self._create_plan(expected_board_before, block, expected_board_after_latest_plan)
            if plan is None:
                # planning was cancelled, or no valid plan was found
                continue

            with self._planning_lock:
                if self._current_plan is None:
                    self._set_current_plan(plan)
                else:
                    self._next_plan = plan

    def _create_plan(self, expected_board_before: Board, block: Block, expected_board_after: Board) -> Plan | None:
        """Create a plan to fit a block into the optimal position given the board state.

        Args:
            expected_board_before: The board state based on which the plan should be created.
            block: The block that a plan should be created for.
            expected_board_after: Output argument that will be set to the expected board state after the plan has been
                executed.

        Returns:
            Plan: The plan to fit the block into the optimal position, consisting of the expected board state before the
                plan is to be executed (for double checking), and the target positioned block (i.e. target position and
                rotation of the block).
        """
        internal_planning_board = Board()
        min_loss: Loss | None = None
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

                loss = Loss.from_board(internal_planning_board)
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_loss_positioned_block = positioned_block
                    expected_board_after.set_from_other(internal_planning_board)

        if min_loss_positioned_block is None:
            return None

        return Plan(expected_board_before, min_loss_positioned_block)

    def get_action(self, board: Board | None = None) -> Action:
        if self._current_plan is None or self._real_board.active_block is None:
            self._active_frame = True
            return Action()

        # the complicated part of the current plan has been executed, now simply quick-drop the block into place
        # (ignore self._active_frame; in this case, we want to hold the button)
        misalignment = self._positioned_blocks_misalignment(
            self._real_board.active_block,
            self._current_plan.target_positioned_block,
        )
        if not misalignment:
            self._active_frame = True
            # only quickdrop if we have already computed the next plan; otherwise we want to buy some time for the next
            # plan to be computed (i.e. slow drop)
            return Action(down=self._next_plan is not None)

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
