import random
from collections.abc import Callable, Iterator
from functools import partial
from typing import NamedTuple, override

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.block import Block, BlockType
from tetris.game_logic.components.board import Board
from tetris.game_logic.components.exceptions import CannotSpawnBlockError
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.core.post_merge.post_merge_rule import PostMergeRule
from tetris.game_logic.rules.core.spawn.synchronized_spawn import SynchronizedSpawning
from tetris.game_logic.rules.messages import PostMergeFinishedMessage, SpawnMessage, SynchronizedSpawnCommandMessage


class SpawnRule(Publisher, Subscriber, Callback, Rule):
    def __init__(self, select_block_fn: Callable[[], Block] = Block.create_random) -> None:
        super().__init__()

        self._select_block_fn = select_block_fn
        self._next_block = self._select_block_fn()

        self._should_spawn = True
        self._synchronized_spawn = False

    @property
    def select_block_fn(self) -> Callable[[], Block]:
        return self._select_block_fn

    @select_block_fn.setter
    def select_block_fn(self, value: Callable[[], Block]) -> None:
        self._select_block_fn = value
        self._next_block = self._select_block_fn()

    @override
    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index

    @override
    def on_game_start(self) -> None:
        self._next_block = self._select_block_fn()
        self._should_spawn = True

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, PostMergeRule | SynchronizedSpawning) and publisher.game_index == self.game_index

    @override
    def add_subscriber(self, subscriber: Subscriber) -> None:
        from tetris.game_logic.rules.special.powerup import PowerupRule

        # make sure the PowerupRule (if it exists) is the first subscriber being notified, to ensure it has the ability
        # to modify the next_block (in-place) before others are notified
        if isinstance(subscriber, PowerupRule):
            self._subscribers.insert(0, subscriber)
        else:
            self._subscribers.append(subscriber)

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if not any(isinstance(p, PostMergeRule) for p in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} is not subscribed to a PostMergeRule: {publishers}"
            raise RuntimeError(msg)

        if any(isinstance(p, SynchronizedSpawning) for p in publishers):
            self._synchronized_spawn = True

    @override
    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, SynchronizedSpawnCommandMessage) or (
            not self._synchronized_spawn and isinstance(message, PostMergeFinishedMessage)
        ):
            self._should_spawn = True

    @override
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        if not self._should_spawn:
            return
        self._should_spawn = False

        try:
            board.spawn(self._next_block)
        except CannotSpawnBlockError as e:
            raise GameOverError from e

        next_block = self._select_block_fn()
        self.notify_subscribers(SpawnMessage(block=self._next_block, next_block=next_block))

        self._next_block = next_block

    @classmethod
    def from_shuffled_bag(cls, seed: int | None = None, bag: list[BlockType] | None = None) -> "SpawnRule":
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
    def truly_random(cls, seed: int | None = None) -> "SpawnRule":
        return cls(select_block_fn=cls.truly_random_selection_fn(seed))

    @staticmethod
    def truly_random_selection_fn(seed: int | None = None) -> Callable[[], Block]:
        return partial(Block.create_random, rng=random.Random(seed))
