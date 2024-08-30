from collections.abc import Iterable, Iterator, Sequence
from random import Random


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
    def __init__(self, items: Iterable[T], seed: int | None = None) -> None:
        self._items = items
        self._rng = Random(seed)

    def __iter__(self) -> Iterator[T]:
        shuffled_items = list(self._items)
        self._rng.shuffle(shuffled_items)
        yield from shuffled_items
