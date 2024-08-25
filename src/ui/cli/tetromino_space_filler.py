import contextlib
import random
import sys
from collections.abc import Generator, Iterator
from itertools import product
from typing import NamedTuple

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray
from skimage import measure

from ansi_extensions import color as colorx
from ansi_extensions import cursor as cursorx
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

        self._total_blocks_to_place = np.sum(~self.space.astype(bool)) // 4
        self._blocks_placed = 0
        self._finished = False
        # self._dead_ends: set[bytes] = set()
        self._unfillable_cell: tuple[int, int] | None = None
        self._smallest_island: NDArray[np.bool] | None = None
        self._i = 0
        self._last_drawn = None

    def draw(self) -> None:
        rd = random.Random()
        if self._last_drawn is None:
            print(cursor.goto(1, 1), end="")
            for row in self.space:
                print(
                    "".join(
                        rd.seed(int(val))  # type: ignore[func-returns-value]
                        or colorx.bg.rgb_truecolor(rd.randrange(50, 256), rd.randrange(50, 256), rd.randrange(50, 256))
                        + "  "
                        + color.fx.reset
                        if val > 0
                        else "  "
                        for val in row
                    )
                )
        else:
            for y, x in np.argwhere(self.space != self._last_drawn):
                print(cursor.goto(y + 1, x * 2 + 1), end="")
                print(
                    rd.seed(int(val))
                    or colorx.bg.rgb_truecolor(rd.randrange(50, 256), rd.randrange(50, 256), rd.randrange(50, 256))
                    + "  "
                    + color.fx.reset
                    if (val := self.space[y, x]) > 0
                    else "  ",
                    end="",
                    flush=True,
                )
                print(cursor.goto(self.space.shape[0] + 1) + cursorx.erase_to_end(""), end="")

        self._last_drawn = self.space.copy()

    def fill(self) -> None:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            self._fill()
        self.draw()

    # TODO consider inlining the 5 functions into one; might make a notable performance difference
    def _fill(
        self, space_view: NDArray[np.int16] | None = None, empty_cell_index_to_fill: tuple[int, int] = (0, 0)
    ) -> None:
        space = space_view if space_view is not None else self.space

        # if self.space.astype(bool).tobytes() in self._dead_ends:
        #     return

        # find the empty cell which is the closes to last_placed_block_center

        # if self._blocks_placed % 10 == 0:
        if np.all(space):
            self._finished = True
            return

        # Prio 1: Try to fill the unfillable cell
        if self._unfillable_cell:
            empty_cell_index_to_fill = self._unfillable_cell
            self._unfillable_cell = None
        # Prio 2: Fill the smallest island (in case the requested index is not inside it, change it)
        elif self._smallest_island is not None and not self._smallest_island[empty_cell_index_to_fill]:
            empty_cell_index_to_fill = self._closest_index_in_area(
                allowed_area=self._smallest_island, index=empty_cell_index_to_fill
            )
        # Prio 3: Select an empty cell as close as possible to the requested cell (the cell itself in case it's empty)
        elif space[empty_cell_index_to_fill] != 0:
            empty_cell_index_to_fill = self._closest_index_in_area(
                allowed_area=self.space == 0, index=empty_cell_index_to_fill
            )

        for next_empty_cell_index_to_fill in self._generate_placements(space, empty_cell_index_to_fill):
            # self._fill(np.rot90(space[1:]) if np.all(space[0]) else space)
            self._fill(empty_cell_index_to_fill=next_empty_cell_index_to_fill)
            if self._finished:
                return

        # remember this one as the cell that could not be filled, then backtrack until one of it's neighbors is removed,
        # hopefully removing the issue that made it unfillable
        if not self._unfillable_cell:
            self._unfillable_cell = empty_cell_index_to_fill
        # self._dead_ends.add(self.space.astype(bool).tobytes())

    def _generate_placements(
        self, space: NDArray[np.int16], cell_to_be_filled_idx: tuple[int, int]
    ) -> Iterator[tuple[int, int]]:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + self._blocks_placed) % len(self.TETROMINOS)]
            quick_return = yield from self._place_tetromino(space, tetromino, cell_to_be_filled_idx)
            if quick_return:
                return

    def _place_tetromino(
        self, space: NDArray[np.int16], tetromino: NDArray[np.bool], cell_to_be_filled_idx: tuple[int, int]
    ) -> Generator[tuple[int, int], None, bool]:
        # for efficiency: make sure only distinct versions of the tetromino are used (avoid duplicates because of
        # symmetry (rotational or axial))
        # TODO: pre-generate all rotations and translations (I guess at import time even)
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
            quick_return = yield from self._place_rotated_tetromino_on_view(
                space, rotated_tetromino, cell_to_be_filled_idx
            )
            if quick_return:
                return True
            hashes.add(tetromino_hash)
        return False

    # def _place_rotated_tetromino(self, tetromino: NDArray[np.bool]) -> Iterator[None]:
    #     space_view = self.space_rotated_transposed_views[
    #         self._blocks_placed % len(self.space_rotated_transposed_views)
    #     ]
    #     yield from self._place_rotated_tetromino_on_view(space_view, tetromino)

    def _place_rotated_tetromino_on_view(
        self, space: NDArray[np.int16], tetromino: NDArray[np.bool], cell_to_be_filled_idx: tuple[int, int]
    ) -> Generator[tuple[int, int], None, bool]:
        for tetromino_cell_idx in np.argwhere(tetromino):
            tetromino_placement_idx = cell_to_be_filled_idx - tetromino_cell_idx

            y, x = tetromino_placement_idx
            height, width = tetromino.shape

            local_space_view = space[y : y + height, x : x + width]
            # In case part of the tetromino placement is partly out of bounds, the slicing syntax above still works, but
            # the shape of the obtained view is not as expected (height, width).
            # We use this as an efficient way of checking whether the placement is partly out of bounds:
            if local_space_view.shape != tetromino.shape:
                # placement partly out of bounds
                continue

            if np.any(np.logical_and(local_space_view, tetromino)):
                continue

            self._blocks_placed += 1
            local_space_view[tetromino] = self._blocks_placed

            filled = False
            if self.space_fillable():
                filled = True
                self._i += 1
                if self._i % 10 == 0:
                    self.draw()
                yield self._get_neighboring_empty_cell_with_most_filled_neighbors_idx(space, tetromino, y, x) or (
                    y + tetromino.shape[0] // 2,
                    x + tetromino.shape[1] // 2,
                )
                self._i += 1

            self._blocks_placed -= 1
            local_space_view[tetromino] = 0
            if filled and self._i % 10 == 0:
                self.draw()

            if self._unfillable_cell and not self._is_close(tetromino, y, x, self._unfillable_cell):
                # We assume that if the block we have just removed during backtracking is not close to the unfillable
                # cell, then this change has likely not made the unfillable block fillable again, thus we don't even
                # try.
                # (Note that not returning would lead to yielding, which would lead to calling _fill, which would
                # lead to trying to fill the unfillable cell)
                return True
        return False

    def _is_close(
        self, tetromino: NDArray[np.bool], y: int, x: int, cell_index: tuple[int, int], distance_threshold: int = 3
    ) -> bool:
        tetromino_idxs = np.argwhere(tetromino) + (y, x)
        tetromino_to_cell_idx_distances = np.sum(np.abs(tetromino_idxs - cell_index), axis=1)
        min_distance = np.min(tetromino_to_cell_idx_distances)
        return min_distance <= distance_threshold

    @staticmethod
    def _is_adjacent(tetromino: NDArray[np.bool], y: int, x: int, cell_index: tuple[int, int]) -> bool:
        cell_y, cell_x = cell_index

        if not TetrominoSpaceFiller._in_or_adjacent_to_bounding_box(tetromino, y, x, cell_y, cell_x):
            return False

        for cell_neighbor_y, cell_neighbor_x in (
            (cell_y - 1, cell_x),
            (cell_y + 1, cell_x),
            (cell_y, cell_x - 1),
            (cell_y, cell_x + 1),
        ):
            local_cell_neighbor_y = cell_neighbor_y - y
            local_cell_neighbor_x = cell_neighbor_x - x
            if (
                0 <= local_cell_neighbor_y < tetromino.shape[0]
                and 0 <= local_cell_neighbor_x < tetromino.shape[1]
                and tetromino[local_cell_neighbor_y, local_cell_neighbor_x]
            ):
                return True

        return False

    @staticmethod
    def _in_or_adjacent_to_bounding_box(tetromino: NDArray[np.bool], y: int, x: int, cell_y: int, cell_x: int) -> bool:
        return y - 1 <= cell_y <= y + tetromino.shape[0] and x - 1 <= cell_x <= x + tetromino.shape[1]

    @staticmethod
    def _get_neighboring_empty_cell_with_most_filled_neighbors_idx(
        space: NDArray[np.int16], tetromino: NDArray[np.bool], y: int, x: int
    ) -> tuple[int, int] | None:
        space_around_tetromino = space[
            max(y - 1, 0) : y + tetromino.shape[0] + 1, max(x - 1, 0) : x + tetromino.shape[1] + 1
        ]
        max_filled_neighbors = 0
        max_filled_neighbors_idx: tuple[int, int] | None = None

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
                return y_in_space, x_in_space

            if filled_neighbors > max_filled_neighbors:
                max_filled_neighbors = filled_neighbors
                max_filled_neighbors_idx = y_in_space, x_in_space

        return max_filled_neighbors_idx

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

    def _first_empty_cell_index(self) -> tuple[int, int]:
        for idx, value in np.ndenumerate(self.space):
            if value == 0:
                return idx
        raise ValueError("Space is full!")

    @staticmethod
    def _closest_index_in_area(allowed_area: NDArray[np.bool], index: tuple[int, int]) -> tuple[int, int]:
        allowed_idxs = np.argwhere(allowed_area)
        distances_to_index = np.abs(allowed_idxs[:, 0] - index[0]) + np.abs(allowed_idxs[:, 1] - index[1])
        return tuple(allowed_idxs[np.argmin(distances_to_index)])

    def space_fillable(self, to_be_placed_tetromino: PositionedTetromino | None = None) -> bool:
        island_map = (self.space == 0).astype(np.uint8)
        space_with_labeled_islands, num_islands = measure.label(island_map, connectivity=1, return_num=True)

        if num_islands <= 1:
            self._smallest_island = None
            return True

        smallest_island_size = self.space.shape[0] * self.space.shape[1]

        for i in range(1, num_islands + 1):
            island = space_with_labeled_islands == i
            island_size = np.sum(island)

            if island_size % 4 != 0:
                self._smallest_island = None
                return False

            if island_size < smallest_island_size:
                smallest_island_size = island_size
                self._smallest_island = island

        return True
