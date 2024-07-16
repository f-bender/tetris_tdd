from math import ceil
from typing import Callable

from exceptions import BaseTetrisException

from game_logic.action_counter import ActionCounter
from game_logic.components import Board
from game_logic.components.block import Block
from game_logic.components.exceptions import CannotDropBlock, CannotSpawnBlock, NoActiveBlock
from game_logic.interfaces.clock import Clock
from game_logic.interfaces.controller import Action, Controller
from game_logic.interfaces.ui import UI


class GameOver(BaseTetrisException):
    pass


class Game:
    MOVE_SINGLE_PRESS_DELAY: int = 15  # for the first n frames of a move button being held, consider it a single press
    MOVE_REPEAT_INTERVAL: int = 4  # if a move button is held longer, perform the action on every nth frame
    ROTATE_REPEAT_INTERVAL: int = 20  # if a rotate button is held, perform the action on every nth frame

    def __init__(
        self,
        ui: UI,
        board: Board,
        controller: Controller,
        clock: Clock,
        initial_frame_interval: int,
        select_new_block_fn: Callable[[], Block] = Block.create_random,
    ) -> None:
        self._ui = ui
        self._board = board
        self._controller = controller
        self._clock = clock
        self._frame_counter = 0
        self._select_new_block_fn = select_new_block_fn  # will be moved into the rule that handles block spawning
        # how many frame pass between e.g. block dropping (lower value = faster gameplay)
        self._frame_interval = initial_frame_interval
        self._quick_drop_interval = ceil(initial_frame_interval / 8)
        self._action_counter = ActionCounter()
        self._last_merge_frame_count: int | None = None

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    def run(self) -> None:
        self.initialize()
        while True:
            self._clock.tick()
            try:
                self.advance_frame(self._controller.get_action())
            except GameOver:
                self._ui.game_over(self._board.as_array())
                return

    def initialize(self) -> None:
        self._ui.initialize(self._board.height, self._board.width)

    def advance_frame(self, action: Action) -> None:
        self._action_counter.update(action)
        self._try_performing_move_rotate()

        quick_drop_held_since = self._action_counter.held_since(Action(quick_drop=True))
        if (
            (quick_drop_held_since == 0 and self._frame_counter % self._frame_interval == 0)
            or (quick_drop_held_since > 0 and (quick_drop_held_since - 1) % self._quick_drop_interval == 0)
        ) and self._board.has_active_block():
            self._drop_or_merge_active_block()
            self._last_merge_frame_count = self._frame_counter

        if (
            self._last_merge_frame_count is None
            or self._frame_counter - self._last_merge_frame_count == self._frame_interval
        ) and not self._board.has_active_block():
            try:
                self._board.spawn(self._select_new_block_fn())
            except CannotSpawnBlock:
                raise GameOver

        self._ui.draw(self._board.as_array())

        self._frame_counter += 1

    def _try_performing_move_rotate(self) -> None:
        try:
            move_left_held_since = self._action_counter.held_since(Action(move_left=True))
            if (
                move_left_held_since == 1
                or move_left_held_since >= self.MOVE_SINGLE_PRESS_DELAY
                and (move_left_held_since - self.MOVE_SINGLE_PRESS_DELAY) % self.MOVE_REPEAT_INTERVAL == 0
            ):
                self._board.try_move_active_block_left()

            move_right_held_since = self._action_counter.held_since(Action(move_right=True))
            if (
                move_right_held_since == 1
                or move_right_held_since >= self.MOVE_SINGLE_PRESS_DELAY
                and (move_right_held_since - self.MOVE_SINGLE_PRESS_DELAY) % self.MOVE_REPEAT_INTERVAL == 0
            ):
                self._board.try_move_active_block_right()

            rotate_left_held_since = self._action_counter.held_since(Action(rotate_left=True))
            if rotate_left_held_since > 0 and (rotate_left_held_since - 1) % self.ROTATE_REPEAT_INTERVAL == 0:
                self._board.try_rotate_active_block_left()

            rotate_right_held_since = self._action_counter.held_since(Action(rotate_right=True))
            if rotate_right_held_since > 0 and (rotate_right_held_since - 1) % self.ROTATE_REPEAT_INTERVAL == 0:
                self._board.try_rotate_active_block_right()
        except NoActiveBlock:
            pass

    def _drop_or_merge_active_block(self) -> None:
        try:
            self._board.drop_active_block()
        except CannotDropBlock:
            self._board.merge_active_block()
