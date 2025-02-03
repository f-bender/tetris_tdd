import random
from collections.abc import Callable, Iterator
from functools import partial
from typing import Protocol

from tetris.game_logic.components.block import Block, BlockType
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotSpawnBlockError
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.rules.core.messages import SpawnMessage


class SpawnStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class SpawnStrategyImpl(Publisher):
    def __init__(self, select_block_fn: Callable[[], Block] = Block.create_random) -> None:
        super().__init__()

        self.select_block_fn = select_block_fn
        self._next_block = self.select_block_fn()

    def apply(self, board: Board) -> None:
        try:
            board.spawn(self._next_block)
        except CannotSpawnBlockError as e:
            raise GameOverError from e

        next_block = self.select_block_fn()
        self.notify_subscribers(SpawnMessage(block=self._next_block, next_block=next_block))

        self._next_block = next_block

    @classmethod
    def from_shuffled_bag(cls, bag: list[Block] | None = None, seed: int | None = None) -> "SpawnStrategyImpl":
        bag = bag or [Block(block_type) for block_type in BlockType]
        rng = random.Random(seed)

        def block_iter() -> Iterator[Block]:
            while True:
                rng.shuffle(bag)
                yield from bag

        block_iterator = block_iter()

        return cls(select_block_fn=lambda: next(block_iterator))

    @classmethod
    def truly_random(cls, seed: int | None = None) -> "SpawnStrategyImpl":
        return cls(select_block_fn=cls.truly_random_select_fn(seed))

    @staticmethod
    def truly_random_select_fn(seed: int | None = None) -> Callable[[], Block]:
        return partial(Block.create_random, rng=random.Random(seed))
