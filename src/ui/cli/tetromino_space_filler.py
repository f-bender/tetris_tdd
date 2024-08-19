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

    def __init__(self, space: NDArray[np.int16]) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled. Zeros will be filled in by tetrominos. -1s are considered holes and will be
                left as is.
        """
        if not set(np.unique(space)).issubset({-1, 0}):
            raise ValueError("Space must consist of -1s and 0s only.")

        # if not self.space_fillable(space):
        #     raise ValueError("Space cannot be filled! Contains at least one island with size not divisible by 4!")

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

        self._total_blocks_to_place = np.sum(~self.space.astype(bool)) // 4
        self._blocks_placed = 0
        self._finished = False
        # self._dead_ends: set[bytes] = set()

    def draw(self) -> None:
        print(cursor.goto(1, 1))
        for row in np.where(self.space == 0, "  ", "XX"):
            print("".join(row))

    def fill(self) -> None:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            self._fill()

    # TODO consider inlining the 5 functions into one; might make a notable performance difference
    def _fill(self, space_view: NDArray[np.int16] | None = None) -> None:
        # time.sleep(0.05)
        space = space_view if space_view is not None else self.space

        # if self.space.astype(bool).tobytes() in self._dead_ends:
        #     return

        # if self._blocks_placed % 10 == 0:
        self.draw()
        if np.all(space):
            self._finished = True
            return

        for _ in self._generate_placements(space):
            self._fill(np.rot90(space[1:]) if np.all(space[0]) else space)
            # self._fill()
            if self._finished:
                return

        # self._dead_ends.add(self.space.astype(bool).tobytes())

    def _generate_placements(self, space: NDArray[np.int16]) -> Iterator[None]:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + self._blocks_placed) % len(self.TETROMINOS)]
            yield from self._place_tetromino(space, tetromino)

    def _place_tetromino(self, space: NDArray[np.int16], tetromino: NDArray[np.bool]) -> Iterator[None]:
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
                yield from self._place_rotated_tetromino_on_view(space, rotated_tetromino)
                hashes.add(tetromino_hash)

    # def _place_rotated_tetromino(self, tetromino: NDArray[np.bool]) -> Iterator[None]:
    #     space_view = self.space_rotated_transposed_views[
    #         self._blocks_placed % len(self.space_rotated_transposed_views)
    #     ]
    #     yield from self._place_rotated_tetromino_on_view(space_view, tetromino)

    def _place_rotated_tetromino_on_view(self, space: NDArray[np.int16], tetromino: NDArray[np.bool]) -> Iterator[None]:
        first_empty_idx = np.array(self._first_empty_cell_index(space))

        for tetromino_cell_idx in np.argwhere(tetromino):
            tetromino_placement_idx = first_empty_idx - tetromino_cell_idx

            if self._placement_out_of_bounds(tetromino_placement_idx, space.shape, tetromino.shape):
                continue

            y, x = tetromino_placement_idx
            height, width = tetromino.shape

            local_space_view = space[y : y + height, x : x + width]

            if np.any(np.logical_and(local_space_view, tetromino)) or not self.space_fillable(
                space, PositionedTetromino((y, x), tetromino)
            ):
                continue

            self._blocks_placed += 1
            local_space_view[tetromino] = self._blocks_placed

            yield

            self._blocks_placed -= 1
            local_space_view[tetromino] = 0

    @staticmethod
    def _first_empty_cell_index(space: NDArray[np.int16]) -> tuple[int, int]:
        for idx, value in np.ndenumerate(space):
            if value == 0:
                return idx
        raise ValueError("Space is full!")

    @staticmethod
    def _placement_out_of_bounds(
        placement_idx: NDArray[np.int_], space_shape: tuple[int, int], tetromino_shape: tuple[int, int]
    ) -> bool:
        return bool(
            np.any(placement_idx < 0) or np.any(placement_idx + np.array(tetromino_shape) > np.array(space_shape))
        )

    # TODO: use check_around argument to make this check more efficient, validating only the area around a newly placed
    # block
    @staticmethod
    def space_fillable(
        space_view: NDArray[np.int16], to_be_placed_tetromino: PositionedTetromino | None = None
    ) -> bool:
        island_map = np.where(space_view == 0, False, True)

        if to_be_placed_tetromino:
            y, x = to_be_placed_tetromino.position
            height, width = to_be_placed_tetromino.tetromino.shape
            island_map[y : y + height, x : x + width] = np.logical_or(
                to_be_placed_tetromino.tetromino,
                island_map[y : y + height, x : x + width],
            )

        return all(island_size % 4 == 0 for island_size in TetrominoSpaceFiller._generate_islands(island_map))

    @staticmethod
    def _generate_islands(island_map: NDArray[np.int16]) -> Iterator[int]:
        for index, cell in np.ndenumerate(island_map):
            if cell == 0:
                yield TetrominoSpaceFiller._flood_fill_island(island_map, index)

    @staticmethod
    def _flood_fill_island(island_map: NDArray[np.int16], index: tuple[int, int]) -> int:
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
