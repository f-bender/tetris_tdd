from math import ceil
from time import perf_counter
from typing import Callable

import ansi
from ansi import cursor
from exceptions import BaseTetrisException

from game_logic.components import Board
from game_logic.components.block import Block
from game_logic.components.exceptions import CannotDropBlock, CannotSpawnBlock, NoActiveBlock
from game_logic.interfaces.clock import Clock
from game_logic.interfaces.controller import Action, Controller
from game_logic.interfaces.ui import UI


class GameOver(BaseTetrisException):
    pass


class Game:
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
        self._quick_drop_interval = ceil(initial_frame_interval / 2)

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
        self._board.spawn_random_block()

    def advance_frame(self, action: Action = Action()) -> None:
        t0 = perf_counter()
        self._try_performing_move_rotate(action)

        if (not action.quick_drop and self._frame_counter % self._frame_interval == 0) or (
            action.quick_drop and self._frame_counter % self._quick_drop_interval == 0
        ):
            if self._board.has_active_block():
                self._drop_or_merge_active_block()
            else:
                try:
                    self._board.spawn(self._select_new_block_fn())
                except CannotSpawnBlock:
                    raise GameOver

        t = perf_counter()
        self._ui.draw(self._board.as_array())
        print(cursor.goto(50, 0) + ansi.color.fx.reset, (perf_counter() - t) * 1e6, "\n", (perf_counter() - t0) * 1e6)

        self._frame_counter += 1

    def _try_performing_move_rotate(self, action: Action) -> None:
        try:
            if action.move_left:
                self._board.try_move_active_block_left()
            if action.move_right:
                self._board.try_move_active_block_right()
            if action.rotate_left:
                self._board.try_rotate_active_block_left()
            if action.rotate_right:
                self._board.try_rotate_active_block_right()
        except NoActiveBlock:
            pass

    def _drop_or_merge_active_block(self) -> None:
        try:
            self._board.drop_active_block()
        except CannotDropBlock:
            self._board.merge_active_block()
