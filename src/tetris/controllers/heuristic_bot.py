"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

import logging
from threading import Lock, Thread
from typing import Final, NamedTuple

import numpy as np

from tetris.game_logic.components.block import Block, Vec
from tetris.game_logic.components.board import Board, PositionedBlock
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

    num_overhanging_cells: int
    average_cell_height: float
    sum_of_adjacent_height_differences: int


class Misalignment(NamedTuple):
    x: int
    rotation: bool

    def __bool__(self) -> bool:
        return self.x != 0 or self.rotation


class HeuristicBotController(Controller, Subscriber):
    def __init__(self, board: Board) -> None:
        """Initializes the bot controller.

        Args:
            board: The board that is actually being played on.
        """
        self._real_board: Final = board

        self._current_block: Block | None = None
        self._next_block: Block | None = None

        self._plan_lock = Lock()
        self._current_plan: Plan | None = None
        self._next_plan: Plan | None = None
        self._cancel_planning_flag: bool = False

        self._active_frame: bool = True

        Thread(target=self._continuously_plan, daemon=True).start()

    def notify(self, message: NamedTuple) -> None:
        if not isinstance(message, SpawnMessage):
            return

        with self._plan_lock:
            self._current_block = message.block
            self._next_block = message.next_block

            if not self._current_plan:
                assert not self._next_plan
                # we have crossed over to a new block, but still don't have a plan for the previous one; tell the
                # planning thread to cancel the planning for the previous block (and start planning for the new one)
                self._cancel_planning_flag = True
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

    # to be executed in separate thread
    # NOTE: planning thread ALWAYS checks _next_plan and starts planning in case it's None.
    # when planning is finished, it checks whether _current_plan is None, and if so, sets _current_plan to the new plan
    # (in this case *it* has to check that the expectation before board matches with the current real board!)
    # Otherwise, it sets _next_plan to the new plan.
    # (this setting should obviously always be done in a thread-safe manner using the lock)
    # NOTE: already when starting to plan, it has to check whether _current_plan is None, and if so use _current_block
    # for the planning, otherwise use _next_block
    # Also, of course it has to take its before_board from the after_board of the _current_plan, and if the current_plan
    # is None, from the real board.
    def _continuously_plan(self) -> None:
        expected_board_after_latest_plan = Board()
        expected_board_after_latest_plan.set_from_other(self._real_board)

    @staticmethod
    def _create_plan(expected_board_before: Board, block: Block, expected_board_after: Board) -> Plan:
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
        # TODO: actual implementation
        return Plan(expected_board_before, PositionedBlock(block, Vec(0, 0)))

    def get_action(self, board: Board | None = None) -> Action:
        if self._current_plan is None or self._real_board.active_block is None:
            self._active_frame = True
            return Action()

        # the complicated part of the current plan has been executed, now simply quick-drop the block into place
        # (ignore self._active_frame; in this case, we want to hold the button)
        misalignment = self._positioned_blocks_misalignment(
            self._current_plan.target_positioned_block,
            self._real_board.active_block,
        )
        if not misalignment:
            self._active_frame = True
            return Action(down=True)

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
            rotation=np.array_equal(block_2.block.cells, block_1.block.cells),
        )
