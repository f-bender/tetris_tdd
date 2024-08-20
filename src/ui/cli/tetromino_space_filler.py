import contextlib
import random
import sys
from collections import deque
from collections.abc import Iterator
from itertools import product
from typing import NamedTuple

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from ansi_extensions import color as colorx
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
        self._3_neighbored_cell_deque: set[tuple[int, int]] = set()
        # self._dead_ends: set[bytes] = set()

    def draw(self) -> None:
        print(cursor.goto(1, 1))
        for row in self.space:
            print(
                "".join(
                    random.seed(int(val))
                    or colorx.bg.rgb_truecolor(
                        random.randrange(25, 256), random.randrange(25, 256), random.randrange(25, 256)
                    )
                    + "  "
                    + color.fx.reset
                    if val > 0
                    else "  "
                    for val in row
                )
            )

    def fill(self) -> None:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            self._fill()
        self.draw()

    # TODO consider inlining the 5 functions into one; might make a notable performance difference
    def _fill(
        self, space_view: NDArray[np.int16] | None = None, empty_cell_index_to_fill: tuple[int, int] | None = None
    ) -> None:
        # time.sleep(0.1)
        space = space_view if space_view is not None else self.space

        # if self.space.astype(bool).tobytes() in self._dead_ends:
        #     return

        # find the empty cell which is the closes to last_placed_block_center

        # if self._blocks_placed % 10 == 0:
        self.draw()
        if np.all(space):
            self._finished = True
            return

        pop = bool(self._3_neighbored_cell_deque)
        empty_cell_index_to_fill = (
            self._3_neighbored_cell_deque.pop()
            if self._3_neighbored_cell_deque
            # TODO: not simply the first empty cell, but the clostest one to the last placed block!
            else self._first_empty_cell_index(space)
        )
        if space[empty_cell_index_to_fill] != 0:
            # print("WEEE WOOO WEEE WOOO")
            empty_cell_index_to_fill = self._first_empty_cell_index(space)

        for next_empty_cell_index_to_fill in self._generate_placements(space, empty_cell_index_to_fill):
            # self._fill(np.rot90(space[1:]) if np.all(space[0]) else space)
            self._fill(empty_cell_index_to_fill=next_empty_cell_index_to_fill)
            if self._finished:
                return

        if pop:
            self._3_neighbored_cell_deque.add(empty_cell_index_to_fill)
        # self._dead_ends.add(self.space.astype(bool).tobytes())

    def _generate_placements(
        self, space: NDArray[np.int16], cell_to_be_filled_idx: tuple[int, int]
    ) -> Iterator[tuple[int, int] | None]:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + self._blocks_placed) % len(self.TETROMINOS)]
            yield from self._place_tetromino(space, tetromino, cell_to_be_filled_idx)

    def _place_tetromino(
        self, space: NDArray[np.int16], tetromino: NDArray[np.bool], cell_to_be_filled_idx: tuple[int, int]
    ) -> Iterator[tuple[int, int] | None]:
        # for efficiency: make sure only distinct versions of the tetromino are used (avoid duplicates because of
        # symmetry (rotational or axial))
        hashes: set[bytes] = set()
        transpose_rotations = list(product((False, True), range(4)))
        for i in range(len(transpose_rotations)):
            transpose, rotations = transpose_rotations[(i + self._blocks_placed) % len(transpose_rotations)]
            rotated_tetromino = np.rot90(tetromino, k=rotations).T if transpose else np.rot90(tetromino, k=rotations)
            # NOTE: tobytes doesn't contain shape information, just the raw flat list of bytes
            # -> Add shape info manually
            tetromino_hash = rotated_tetromino.tobytes() + bytes(rotated_tetromino.shape[0])
            if tetromino_hash in hashes:
                continue
            yield from self._place_rotated_tetromino_on_view(space, rotated_tetromino, cell_to_be_filled_idx)
            hashes.add(tetromino_hash)

    # def _place_rotated_tetromino(self, tetromino: NDArray[np.bool]) -> Iterator[None]:
    #     space_view = self.space_rotated_transposed_views[
    #         self._blocks_placed % len(self.space_rotated_transposed_views)
    #     ]
    #     yield from self._place_rotated_tetromino_on_view(space_view, tetromino)

    def _place_rotated_tetromino_on_view(
        self, space: NDArray[np.int16], tetromino: NDArray[np.bool], cell_to_be_filled_idx: tuple[int, int]
    ) -> Iterator[tuple[int, int] | None]:
        for tetromino_cell_idx in np.argwhere(tetromino):
            tetromino_placement_idx = cell_to_be_filled_idx - tetromino_cell_idx

            if self._placement_out_of_bounds(tetromino_placement_idx, space.shape, tetromino.shape):
                continue

            y, x = tetromino_placement_idx
            height, width = tetromino.shape

            local_space_view = space[y : y + height, x : x + width]

            if np.any(np.logical_and(local_space_view, tetromino)):
                continue

            self._blocks_placed += 1
            local_space_view[tetromino] = self._blocks_placed

            if not self._placement_created_new_island(space, tetromino, y, x) or self.space_fillable(space):
                # MAYBE: in case of a new island being created, make sure the smallest of the islands is the one from
                # which the next cell to fill is taken? (might be obsolete once we simply make sure all 3-neighbored
                # cells are taken care of ASAP)
                cells = set(self._get_neighboring_empty_cells_with_3_filled_neighbors_idxs(space, tetromino, y, x))
                self._3_neighbored_cell_deque |= cells
                yield
                self._3_neighbored_cell_deque -= cells

            self._blocks_placed -= 1
            local_space_view[tetromino] = 0

    @staticmethod
    def _get_neighboring_empty_cells_with_3_filled_neighbors_idxs(
        space: NDArray[np.int16], tetromino: NDArray[np.bool], y: int, x: int
    ) -> Iterator[tuple[int, int]]:
        space_around_tetromino = space[
            max(y - 1, 0) : y + tetromino.shape[0] + 1, max(x - 1, 0) : x + tetromino.shape[1] + 1
        ]
        # max_filled_neighbors = 0
        # max_filled_neighbors_idx: tuple[int, int] | None = None

        windows = np.lib.stride_tricks.sliding_window_view(space_around_tetromino, tetromino.shape)
        for (offset_y, offset_x, window_y, window_x), value in np.ndenumerate(windows):
            if value != 0 or not tetromino[window_y, window_x]:
                continue
            # offset is the index of the sliding window, counting from the top left of the space_around_tetromino
            # I am interested in the offset from the *center* window, which is a difference of 1:
            if y > 0:
                offset_y -= 1
            if x > 0:
                offset_x -= 1
            if bool(offset_y) == bool(offset_x):
                # discard diagonal movement or not movement at all
                continue

            y_in_space = y + offset_y + window_y
            x_in_space = x + offset_x + window_x

            filled_neighbors = TetrominoSpaceFiller._count_filled_neighbors(space, x_in_space, y_in_space)

            if filled_neighbors == 3:
                # 3 is the maximum possible -> immediately return
                yield y_in_space, x_in_space

        #     if filled_neighbors > max_filled_neighbors:
        #         max_filled_neighbors = filled_neighbors
        #         max_filled_neighbors_idx = y_in_space, x_in_space

        # return max_filled_neighbors_idx

    @staticmethod
    def _placement_created_new_island(space: NDArray[np.int16], tetromino: NDArray[np.bool], y: int, x: int) -> bool:
        space_around_tetromino = space[
            max(y - 1, 0) : y + tetromino.shape[0] + 1, max(x - 1, 0) : x + tetromino.shape[1] + 1
        ]
        tetromino_offset_in_space_around_tetromino_y = 0 if y == 0 else 1
        tetromino_offset_in_space_around_tetromino_x = 0 if x == 0 else 1
        tetromino_outline = np.ones_like(space_around_tetromino, dtype=np.bool)

        windows = np.lib.stride_tricks.sliding_window_view(space_around_tetromino, tetromino.shape)
        for (offset_y, offset_x, window_y, window_x), value in np.ndenumerate(windows):
            if value != 0 or not tetromino[window_y, window_x]:
                continue

            y_in_space_around_tetromino = offset_y + window_y
            x_in_space_around_tetromino = offset_x + window_x
            tetromino_outline[y_in_space_around_tetromino, x_in_space_around_tetromino] = False

        # tetromino_outline[
        #     tetromino_offset_in_space_around_tetromino_y : tetromino_offset_in_space_around_tetromino_y
        #     + tetromino.shape[0],
        #     tetromino_offset_in_space_around_tetromino_x : tetromino_offset_in_space_around_tetromino_x
        #     + tetromino.shape[1],
        # ] = np.logical_and(
        #     tetromino_outline[
        #         tetromino_offset_in_space_around_tetromino_y : tetromino_offset_in_space_around_tetromino_y
        #         + tetromino.shape[0],
        #         tetromino_offset_in_space_around_tetromino_x : tetromino_offset_in_space_around_tetromino_x
        #         + tetromino.shape[1],
        #     ],
        #     ~tetromino,
        # )
        tetromino_outline[
            tetromino_offset_in_space_around_tetromino_y : tetromino_offset_in_space_around_tetromino_y
            + tetromino.shape[0],
            tetromino_offset_in_space_around_tetromino_x : tetromino_offset_in_space_around_tetromino_x
            + tetromino.shape[1],
        ][tetromino] = True

        return len(list(TetrominoSpaceFiller._generate_islands(island_map=tetromino_outline))) > 1

    @staticmethod
    def _count_filled_neighbors(space: NDArray[np.int16], x: int, y: int) -> int:
        return sum(
            (
                not 0 <= neighbor_y < space.shape[0]
                or not 0 <= neighbor_x < space.shape[1]
                or space[neighbor_y, neighbor_x] != 0
            )
            for neighbor_y, neighbor_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        )

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
