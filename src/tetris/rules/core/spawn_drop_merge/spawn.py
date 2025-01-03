from collections.abc import Callable
from typing import Protocol

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotSpawnBlockError
from tetris.game_logic.game import GameOverError


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
        except CannotSpawnBlockError as e:
            raise GameOverError from e
