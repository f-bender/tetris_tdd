from tetris.tetromino_space_filler.offset_iterables import (
    CyclingOffsetIterable,
    RandomOffsetIterable,
    RandomOrderIterable,
)


def test_cycling_offset_iterable() -> None:
    iterable = CyclingOffsetIterable(range(10))

    # preserve  contained elements and their order, cycle through starting index
    assert list(iterable) == list(range(10))
    assert list(iterable) == list(range(1, 10)) + list(range(1))
    assert list(iterable) == list(range(2, 10)) + list(range(2))
    assert list(iterable) == list(range(3, 10)) + list(range(3))
    assert list(iterable) == list(range(4, 10)) + list(range(4))
    assert list(iterable) == list(range(5, 10)) + list(range(5))
    assert list(iterable) == list(range(6, 10)) + list(range(6))
    assert list(iterable) == list(range(7, 10)) + list(range(7))
    assert list(iterable) == list(range(8, 10)) + list(range(8))
    assert list(iterable) == list(range(9, 10)) + list(range(9))

    assert list(iterable) == list(range(10))
    assert list(iterable) == list(range(1, 10)) + list(range(1))
    assert list(iterable) == list(range(2, 10)) + list(range(2))


def test_random_offset_iterable() -> None:
    iterable = RandomOffsetIterable(range(10))

    for _ in range(100):
        # start at a random index but preserve contained elements and their order
        assert list(iterable) in [list(range(offset, 10)) + list(range(offset)) for offset in range(10)]


def test_random_order_iterable() -> None:
    iterable = RandomOrderIterable(range(10))

    for _ in range(100):
        # randomize order but preserve contained elements
        assert set(iterable) == set(range(10))
