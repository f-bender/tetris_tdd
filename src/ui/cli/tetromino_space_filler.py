import contextlib
import sys
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from ansi import cursor
from numpy.typing import NDArray

from game_logic.components.block import Block, BlockType


@contextlib.contextmanager
def ensure_sufficient_recursion_depth(depth: int) -> Iterator[None]:
    prev_depth = sys.getrecursionlimit()
    sys.setrecursionlimit(max(depth, prev_depth))
    try:
        yield
    finally:
        sys.setrecursionlimit(prev_depth)


class PositionedTetromino(NamedTuple):
    position: tuple[int, int]
    tetromino: NDArray[np.bool]


class Island(NamedTuple):
    cells: set[tuple[int, int]]

    @property
    def bounding_box(self) -> tuple[slice, slice]:
        min_x = min(x for x, _ in self.cells)
        max_x = max(x for x, _ in self.cells)
        min_y = min(y for _, y in self.cells)
        max_y = max(y for _, y in self.cells)
        return slice(min_x, max_x + 1), slice(min_y, max_y + 1)


class TetrominoSpaceFiller:
    TETROMINOS = tuple(
        Block(block_type).actual_cells
        for block_type in (
            # J and Z are omitted since transposing, i.e. mirroring is done in the process of filling
            BlockType.T,
            BlockType.O,
            BlockType.I,
            BlockType.L,
            BlockType.S,
        )
    )
    STACK_FRAMES_SAFETY_MARGIN = 50

    def __init__(self, space: NDArray[np.int16], blocks_placed: int = 0, to_be_filled_value: int = -1) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled. Zeros will be filled in by tetrominos. -1s are considered holes and will be
                left as is.
        """
        if to_be_filled_value >= 0:
            raise ValueError(
                "0 is used for holes, positive numbers will be filled in by tetrominos. "
                "`to_be_filled_value` must be < 0 and should be left at -1 when called from outside."
            )
        self._to_be_filled_value = to_be_filled_value

        if self._to_be_filled_value == -1 and not set(np.unique(space)).issubset({-1, 0}):
            raise ValueError("Space must consist of -1s and 0s only.")

        self.space = space

        self.space_rotated_transposed_views = (
            self.space,
            # np.rot90(self.space, k=1).T,
            # np.rot90(self.space, k=1),
            # np.rot90(self.space, k=2).T,
            # np.rot90(self.space, k=2),
            # np.rot90(self.space, k=3).T,
            # np.rot90(self.space, k=3),
            # self.space.T,
        )

        self._total_blocks_to_place = np.sum(self.space == self._to_be_filled_value) // 4
        self._blocks_placed = blocks_placed
        self._finished = False
        self._dead_ends: set[bytes] = set()

    def draw(self) -> None:
        print(cursor.goto(1, 1))
        for row in np.where(self.space < 0, "  ", "XX"):
            print("".join(row))

    def fill(self) -> bool:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            return self._fill()

    # TODO consider inlining the 5 functions into one; might make a notable performance difference
    def _fill(self) -> bool:
        if (self.space >= 0).tobytes() in self._dead_ends:
            return False

        self.draw()
        if np.all(self.space >= 0):
            self._finished = True
            return True

        for island_size in self.flood_fill_islands():
            self._to_be_filled_value -= 1

            if island_size % 4 != 0:
                self.space[self.space == self._to_be_filled_value] += 1
                self._to_be_filled_value += 1
                return False

            for _ in self._generate_placements():
                self._fill()
                if self._finished:
                    return True

            self.space[self.space == self._to_be_filled_value] += 1
            self._to_be_filled_value += 1

        self._dead_ends.add((self.space >= 0).tobytes())
        return False

    def flood_fill_islands(self) -> Iterator[int]:
        island_idxs_sizes: list[tuple[tuple[int, int], int]] = []
        for index, cell in np.ndenumerate(self.space):
            if cell == self._to_be_filled_value:
                island_idxs_sizes.append((index, self.flood_fill_island(index)))

        self.space[self.space == self._to_be_filled_value - 1] += 1
        for idx, _ in sorted(island_idxs_sizes, key=lambda x: x[1], reverse=True):
            yield self.flood_fill_island(idx)

    def flood_fill_island(self, index: tuple[int, int]) -> int:
        island_size = 0
        to_visit = {index}

        while to_visit:
            y, x = to_visit.pop()
            island_size += 1
            self.space[y, x] = self._to_be_filled_value - 1
            to_visit |= {
                idx
                for idx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
                if TetrominoSpaceFiller._in_bounds(self.space.shape, idx)
                and self.space[idx] == self._to_be_filled_value
            }

        return island_size

    def _generate_placements(self) -> Iterator[None]:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + self._blocks_placed) % len(self.TETROMINOS)]
            yield from self._place_tetromino(tetromino)

    def _place_tetromino(self, tetromino: NDArray[np.bool]) -> Iterator[None]:
        # for efficiency: make sure only distinct versions of the tetromino are used (avoid duplicates because of
        # symmetry (rotational or axial))
        hashes: set[bytes] = set()
        for transpose in (False, True):
            for k in range(4):
                rotated_tetromino = np.rot90(tetromino, k=k).T if transpose else np.rot90(tetromino, k=k)
                # NOTE: tobytes doesn't contain shape information, just the raw flat list of bytes
                # -> Add shape info manually
                tetromino_hash = rotated_tetromino.tobytes() + bytes(rotated_tetromino.shape[0])
                if tetromino_hash in hashes:
                    continue
                yield from self._place_rotated_tetromino(rotated_tetromino)
                hashes.add(tetromino_hash)

    def _place_rotated_tetromino(self, tetromino: NDArray[np.bool]) -> Iterator[None]:
        space_view = self.space_rotated_transposed_views[self._blocks_placed % len(self.space_rotated_transposed_views)]
        yield from self._place_rotated_tetromino_on_view(space_view, tetromino)

    def _place_rotated_tetromino_on_view(
        self, space_view: NDArray[np.int16], tetromino: NDArray[np.bool]
    ) -> Iterator[None]:
        windows = np.lib.stride_tricks.sliding_window_view(space_view, tetromino.shape)
        for window_index in np.ndindex(windows.shape[:2]):
            if np.any(
                np.logical_and(windows[window_index] != self._to_be_filled_value, tetromino)
            ) or not self.space_fillable(
                space_view, to_be_placed_tetromino=PositionedTetromino(window_index, tetromino)
            ):
                continue

            y, x = window_index
            height, width = tetromino.shape

            self._blocks_placed += 1
            space_view[y : y + height, x : x + width] = np.where(
                tetromino,
                self._blocks_placed,
                space_view[y : y + height, x : x + width],
            )

            yield

            self._blocks_placed -= 1
            space_view[y : y + height, x : x + width] = np.where(
                tetromino,
                self._to_be_filled_value,
                space_view[y : y + height, x : x + width],
            )
            pass

    # TODO: use check_around argument to make this check more efficient, validating only the area around a newly placed
    # block
    @staticmethod
    def space_fillable(
        space_view: NDArray[np.int16], to_be_placed_tetromino: PositionedTetromino | None = None
    ) -> bool:
        island_map = np.where(space_view < 0, False, True)

        if to_be_placed_tetromino:
            y, x = to_be_placed_tetromino.position
            height, width = to_be_placed_tetromino.tetromino.shape
            island_map[y : y + height, x : x + width] = np.logical_or(
                to_be_placed_tetromino.tetromino,
                island_map[y : y + height, x : x + width],
            )

        return all(island_size % 4 == 0 for island_size in TetrominoSpaceFiller._generate_islands(island_map))

    @staticmethod
    def _generate_islands(island_map: NDArray[np.bool]) -> Iterator[int]:
        for index, cell in np.ndenumerate(island_map):
            if not cell:
                yield TetrominoSpaceFiller._flood_fill_island(island_map, index)

    @staticmethod
    def _flood_fill_island(island_map: NDArray[np.bool], index: tuple[int, int]) -> int:
        island_size = 0
        to_visit = [index]

        while to_visit:
            y, x = to_visit.pop()
            if not island_map[y, x]:
                island_size += 1
                island_map[y, x] = True
                to_visit.extend(
                    idx
                    for idx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
                    if TetrominoSpaceFiller._in_bounds(island_map.shape, idx) and not island_map[idx]
                )

        return island_size

    @staticmethod
    def _in_bounds(shape: tuple[int, int], index: tuple[int, int]) -> bool:
        return 0 <= index[0] < shape[0] and 0 <= index[1] < shape[1]
