from collections.abc import Collection, Iterator, Sequence
from copy import copy
from random import Random
from typing import Self


class CyclingOffsetIterable[T]:
    def __init__(self, items: Sequence[T], initial_offset: int = 0) -> None:
        self._items = items
        self._offset = initial_offset

    def __iter__(self) -> Iterator[T]:
        self._offset += 1
        for i in range(self._offset - 1, self._offset + len(self) - 1):
            yield self._items[i % len(self)]

    def __len__(self) -> int:
        return len(self._items)


class RandomOffsetIterable[T]:
    def __init__(self, items: Sequence[T], seed: int | None = None) -> None:
        self._items = items
        self._rng = Random(seed)

    def __iter__(self) -> Iterator[T]:
        offset = self._rng.randrange(len(self))
        for i in range(offset, offset + len(self)):
            yield self._items[i % len(self)]

    def __len__(self) -> int:
        return len(self._items)


class RandomOrderIterable[T]:
    def __init__(self, items: Collection[T], seed: int | None = None) -> None:
        self._items = items
        self._rng = Random(seed)

    def __iter__(self) -> Iterator[T]:
        shuffled_items = list(self._items)
        self._rng.shuffle(shuffled_items)
        yield from shuffled_items


class OnceResettable[T]:
    def __init__(self, it: Iterator[T]) -> None:
        self.it = it
        self.it_copy = copy(it)

    def reset(self) -> None:
        self.it = self.it_copy

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        return next(self.it)
