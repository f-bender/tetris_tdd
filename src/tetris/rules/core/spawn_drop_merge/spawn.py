from collections.abc import Callable
from typing import NamedTuple, Protocol

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotSpawnBlockError
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.rule import Publisher


class SpawnStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class SpawnMessage(NamedTuple):
    block: Block
    next_block: Block


class SpawnStrategyImpl(Publisher):
    def __init__(self, select_block_fn: Callable[[], Block] = Block.create_random) -> None:
        super().__init__()

        self._select_block_fn = select_block_fn
        self._next_block = self._select_block_fn()

    def set_select_block_fn(self, select_block_fn: Callable[[], Block]) -> None:
        self._select_block_fn = select_block_fn

    def apply(self, board: Board) -> None:
        try:
            board.spawn(self._next_block)
        except CannotSpawnBlockError as e:
            raise GameOverError from e

        next_block = self._select_block_fn()
        self.notify_subscribers(SpawnMessage(block=self._next_block, next_block=next_block))

        self._next_block = next_block
