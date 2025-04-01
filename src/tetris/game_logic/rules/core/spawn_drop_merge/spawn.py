import random
from collections.abc import Callable, Iterator
from functools import partial
from typing import Protocol

from tetris.game_logic.components.block import Block, BlockType
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotSpawnBlockError
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.pub_sub import Publisher
from tetris.game_logic.rules.messages import SpawnMessage


class SpawnStrategy(Protocol):
    def apply(self, board: Board) -> None: ...


class SpawnStrategyImpl(Publisher):
    def __init__(self, select_block_fn: Callable[[], Block] = Block.create_random) -> None:
        super().__init__()

        self._select_block_fn = select_block_fn
        self._next_block = self._select_block_fn()

    @property
    def select_block_fn(self) -> Callable[[], Block]:
        return self._select_block_fn

    @select_block_fn.setter
    def select_block_fn(self, value: Callable[[], Block]) -> None:
        self._select_block_fn = value
        self._next_block = self._select_block_fn()

    def apply(self, board: Board) -> None:
        try:
            board.spawn(self._next_block)
        except CannotSpawnBlockError as e:
            raise GameOverError from e

        next_block = self._select_block_fn()
        self.notify_subscribers(SpawnMessage(block=self._next_block, next_block=next_block))

        self._next_block = next_block

    @classmethod
    def from_shuffled_bag(cls, seed: int | None = None, bag: list[BlockType] | None = None) -> "SpawnStrategyImpl":
        return cls(select_block_fn=cls.from_shuffled_bag_selection_fn(seed=seed, bag=bag))

    @staticmethod
    def from_shuffled_bag_selection_fn(
        seed: int | None = None, bag: list[BlockType] | None = None
    ) -> Callable[[], Block]:
        bag = bag or list(BlockType)
        rng = random.Random(seed)

        def block_iter() -> Iterator[Block]:
            while True:
                rng.shuffle(bag)
                for block_type in bag:
                    yield Block(block_type)

        block_iterator = block_iter()

        return lambda: next(block_iterator)

    @classmethod
    def truly_random(cls, seed: int | None = None) -> "SpawnStrategyImpl":
        return cls(select_block_fn=cls.truly_random_selection_fn(seed))

    @staticmethod
    def truly_random_selection_fn(seed: int | None = None) -> Callable[[], Block]:
        return partial(Block.create_random, rng=random.Random(seed))
