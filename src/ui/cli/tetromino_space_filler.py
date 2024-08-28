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
    CLOSE_DISTANCE_THRESHOLD = 4

    def __init__(self, space: NDArray[np.int32]) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled. Zeros will be filled in by tetrominos. -1s are considered holes and will be
                left as is.
        """
        if not set(np.unique(space)).issubset({-1, 0}):
            raise ValueError("Space must consist of -1s and 0s only.")

        self.space = space
        cells_to_fill = np.sum(~self.space.astype(bool))

        if cells_to_fill % 4 != 0 or not self.space_fillable():
            raise ValueError("Space cannot be filled! Contains at least one island with size not divisible by 4!")

        self._total_blocks_to_place = cells_to_fill // 4
        self._num_blocks_placed = 0
        self._finished = False
        # self._dead_ends: set[bytes] = set()
        self._unfillable_cell: tuple[int, int] | None = None
        self._smallest_island: NDArray[np.bool] | None = None
        self._i = 0
        self._last_drawn = None

    @property
    def total_block_to_place(self) -> int:
        return self._total_blocks_to_place

    @property
    def num_blocks_placed(self) -> int:
        return self._num_blocks_placed

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

    def fill(self, start_position: tuple[int, int] = (0, 0)) -> None:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            self._fill(cell_to_fill_position=start_position)
        self.draw()

    # TODO consider inlining the 5 functions into one; might make a notable performance difference
    def _fill(self, cell_to_fill_position: tuple[int, int] = (0, 0)) -> None:
        if np.all(self.space):
            self._finished = True
            return

        # Prio 1: Try to fill the unfillable cell
        if self._unfillable_cell:
            cell_to_fill_position = self._unfillable_cell
            self._unfillable_cell = None
        # Prio 2: Fill the smallest island (in case the requested position is not inside it, change it)
        elif self._smallest_island is not None and not self._smallest_island[cell_to_fill_position]:
            cell_to_fill_position = self._closest_position_in_area(
                allowed_area=self._smallest_island, position=cell_to_fill_position
            )
        # Prio 3: Select an empty cell as close as possible to the requested cell (the cell itself in case it's empty)
        elif self.space[cell_to_fill_position] != 0:
            cell_to_fill_position = self._closest_position_in_area(
                allowed_area=self.space == 0, position=cell_to_fill_position
            )

        # whether we used it or not, unset self._smallest_island as it will be invalid after the next placement
        self._smallest_island = None

        for next_cell_to_fill_position in self._generate_placements(cell_to_fill_position):
            self._fill(cell_to_fill_position=next_cell_to_fill_position)
            if self._finished:
                return

        # at this point we either have tried everything to fill `cell_to_fill_position` but were unsuccessful,
        # or are in the process of fast backtracking because we have encountered an unfillable cell deeper down the
        # stack

        # in case we are not already in the process of fast backtracking to a cell we were unable to fill
        if not self._unfillable_cell:
            # remember this cell as the cell that could not be filled, then fast backtrack until a cell close to it is
            # removed, hopefully removing the issue that made it unfillable
            self._unfillable_cell = cell_to_fill_position

    def _generate_placements(self, cell_to_fill_position: tuple[int, int]) -> Iterator[tuple[int, int]]:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + self._num_blocks_placed) % len(self.TETROMINOS)]
            quick_return = yield from self._place_tetromino(tetromino, cell_to_fill_position)
            if quick_return:
                return

    def _place_tetromino(
        self, tetromino: NDArray[np.bool], cell_to_fill_position: tuple[int, int]
    ) -> Generator[tuple[int, int], None, bool]:
        # for efficiency: make sure only distinct versions of the tetromino are used (avoid duplicates because of
        # symmetry (rotational or axial))
        # TODO: pre-generate all rotations and translations (I guess at import time even)
        # TODO: i.e. have one big list of all rotations of all tetrominos that we then iterate over, trying to fill the
        # cell
        hashes: set[bytes] = set()
        transpose_rotations = list(product((False, True), range(4)))
        for i in range(len(transpose_rotations)):
            transpose, rotations = transpose_rotations[(i + self._num_blocks_placed) % len(transpose_rotations)]
            rotated_tetromino = np.rot90(tetromino, k=rotations).T if transpose else np.rot90(tetromino, k=rotations)
            # NOTE: tobytes doesn't contain shape information, just the raw flat list of bytes
            # -> Add shape info manually
            tetromino_hash = rotated_tetromino.tobytes() + bytes(rotated_tetromino.shape[0])
            if tetromino_hash in hashes:
                continue
            quick_return = yield from self._place_rotated_tetromino_on_view(rotated_tetromino, cell_to_fill_position)
            if quick_return:
                return True
            hashes.add(tetromino_hash)
        return False

    def _place_rotated_tetromino_on_view(
        self, tetromino: NDArray[np.bool], cell_to_fill_position: tuple[int, int]
    ) -> Generator[tuple[int, int], None, bool]:
        for cell_position_in_tetromino in np.argwhere(tetromino):
            tetromino_position = cell_to_fill_position - cell_position_in_tetromino

            local_space_view = self.space[
                tetromino_position[0] : tetromino_position[0] + tetromino.shape[0],
                tetromino_position[1] : tetromino_position[1] + tetromino.shape[1],
            ]
            # In case part of the tetromino placement is partly out of bounds, the slicing syntax above still works, but
            # the shape of the obtained view is not as expected (height, width).
            # We use this as an efficient way of checking whether the placement is partly out of bounds:
            if local_space_view.shape != tetromino.shape:
                # placement partly out of bounds
                continue

            if np.any(np.logical_and(local_space_view, tetromino)):
                continue

            self._num_blocks_placed += 1
            local_space_view[tetromino] = self._num_blocks_placed

            filled = False
            space_around_tetromino = self.space[
                max(tetromino_position[0] - 1, 0) : tetromino_position[0] + tetromino.shape[0] + 1,
                max(tetromino_position[1] - 1, 0) : tetromino_position[1] + tetromino.shape[1] + 1,
            ]
            # For efficiency reasons, we first check only the area directly around the tetromino.
            # If the empty space in that area is fully connected, then the tetromino placement has not created a new
            # island which means that the space_fillable check can be skipped. (This is usually the case.)
            # Only when the current placement creates new islands of empty space do we have to check that these islands
            # are all still fillable (have a size divisible by 4).
            if not self._empty_space_has_multiple_islands(space_around_tetromino) or self.space_fillable():
                filled = True
                self._i += 1
                if self._i % 10 == 0:
                    self.draw()
                next_cell_to_fill_position = self._get_neighboring_empty_cell_with_least_empty_neighbors_position(
                    PositionedTetromino(tetromino_position, tetromino)
                )
                if next_cell_to_fill_position is None:
                    # no neighbors to fill next: yield the center of the current tetromino and let _fill find the
                    # closest empty cell to it
                    next_cell_to_fill_position = (
                        tetromino_position[0] + tetromino.shape[0] // 2,
                        tetromino_position[1] + tetromino.shape[1] // 2,
                    )
                    # in this case we should also make sure that self._smallest_island is set because the next cell to
                    # be filled should be inside the now smallest island (in case there are multiple islands)
                    if self._smallest_island is None:
                        # note that space_fillable sets the value of _smallest_island
                        self.space_fillable()

                yield next_cell_to_fill_position

                self._i += 1

            self._num_blocks_placed -= 1
            local_space_view[tetromino] = 0
            if filled and self._i % 10 == 0:
                self.draw()

            if self._unfillable_cell and not self._is_close(
                PositionedTetromino(tetromino_position, tetromino), self._unfillable_cell
            ):
                # We assume that if the block we have just removed during backtracking is not close to the unfillable
                # cell, then this change has likely not made the unfillable block fillable again, thus we don't even
                # try.
                # (Note that not returning would lead to yielding, which would lead to calling _fill, which would
                # lead to trying to fill the unfillable cell)
                return True
        return False

    def _is_close(self, positioned_tetromino: PositionedTetromino, cell_position: tuple[int, int]) -> bool:
        tetromino_cell_positions = np.argwhere(positioned_tetromino.tetromino) + positioned_tetromino.position
        tetromino_to_cell_distances = np.sum(np.abs(tetromino_cell_positions - cell_position), axis=1)
        min_distance = np.min(tetromino_to_cell_distances)
        return min_distance <= self.CLOSE_DISTANCE_THRESHOLD

    def _get_neighboring_empty_cell_with_least_empty_neighbors_position(
        self, positioned_tetromino: PositionedTetromino
    ) -> tuple[int, int] | None:
        # TODO create only once and reuse (e.g. "Selector" class)
        tetromino_cell_idxs__neighbor_offsets = list(product(range(4), ((-1, 0), (1, 0), (0, -1), (0, 1))))

        min_empty_neighbors = 4
        min_empty_neighbors_position: tuple[int, int] | None = None

        tetromino_cell_positions = np.argwhere(positioned_tetromino.tetromino) + positioned_tetromino.position
        assert len(tetromino_cell_positions) == 4

        for i in range(len(tetromino_cell_idxs__neighbor_offsets)):
            tetromino_cell_idx, neighbor_offset = tetromino_cell_idxs__neighbor_offsets[
                (i + self._num_blocks_placed) % len(tetromino_cell_idxs__neighbor_offsets)
            ]
            neighbor_position = tetromino_cell_positions[tetromino_cell_idx] + neighbor_offset

            if (
                not 0 <= neighbor_position[0] < self.space.shape[0]
                or not 0 <= neighbor_position[1] < self.space.shape[1]
                or self.space[*neighbor_position] != 0
            ):
                continue

            empty_neighbors = self._count_empty_neighbors(neighbor_position)
            if empty_neighbors == 1:
                return tuple(neighbor_position)

            if empty_neighbors < min_empty_neighbors:
                min_empty_neighbors = empty_neighbors
                min_empty_neighbors_position = tuple(neighbor_position)

        return min_empty_neighbors_position

    def _count_empty_neighbors(self, position: tuple[int, int]) -> int:
        # fmt: off
        return sum(
            (
                0 <= neighbor_y < self.space.shape[0] and
                0 <= neighbor_x < self.space.shape[1] and
                self.space[neighbor_y, neighbor_x] == 0
            )
            for neighbor_y, neighbor_x in (
                (position[0] - 1, position[1]    ),
                (position[0] + 1, position[1]    ),
                (position[0]    , position[1] - 1),
                (position[0]    , position[1] + 1),
            )
        )
        # fmt: on

    @staticmethod
    def _closest_position_in_area(allowed_area: NDArray[np.bool], position: tuple[int, int]) -> tuple[int, int]:
        allowed_positions = np.argwhere(allowed_area)
        distances_to_position = np.abs(allowed_positions[:, 0] - position[0]) + np.abs(
            allowed_positions[:, 1] - position[1]
        )
        return tuple(allowed_positions[np.argmin(distances_to_position)])

    def space_fillable(self, to_be_placed_tetromino: PositionedTetromino | None = None) -> bool:
        # TODO document that this determines the smallest island as well as checking if the remaining space is fillable
        # (ideally in the method name, but at least in a docstring)
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

    @staticmethod
    def _empty_space_has_multiple_islands(space: NDArray[np.int32]) -> bool:
        island_map = (space == 0).astype(np.uint8)
        return measure.label(island_map, connectivity=1, return_num=True)[1] > 1
