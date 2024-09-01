import contextlib
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray
from skimage import measure

from game_logic.components.block import Block, BlockType
from ui.cli.offset_iterables import CyclingOffsetIterable, RandomOffsetIterable


@contextlib.contextmanager
def ensure_sufficient_recursion_depth(depth: int) -> Iterator[None]:
    prev_depth = sys.getrecursionlimit()
    sys.setrecursionlimit(max(depth, prev_depth))
    try:
        yield
    finally:
        sys.setrecursionlimit(prev_depth)


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

    def __init__(
        self,
        space: NDArray[np.int32],
        use_rng: bool = True,
        rng_seed: int | None = None,
        top_left_tendency: bool = False,
        space_updated_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled. Zeros will be filled in by tetrominos. -1s are considered holes and will be
                left as is.
            use_rng: Whether to use randomness in the selection of tetromino, selection of tetromino rotation/transpose,
                and selection of neighboring spot to fill next.
            rng_seed: Optional seed to use for all RNG.
            top_left_tendency: Whether to have a slight bias towards selecting spots top left of the last placed
                tetromino as the next spot to be filled. This causes a tendency for the algorithm to gravitate towards
                the top left of the space. This makes the filling up more predictable, but also reduces the likelihood
                of large backtracks becoming necessary because the "surface area" of tetrominos tends to be smaller.
                If True, selection of neighboring spot to fill next will not be random, even if use_rng is True.
            space_updated_callback: Callback function which is called each time after the space has been updated (a
                tetromino has been placed or removed). This effectively temporarily hands control back to the user of
                this class, letting it act one the latest space update (e.g. for drawing).
        """
        if not use_rng and rng_seed is not None:
            raise ValueError("rng_seed should only be set when use_rng is True")

        if not set(np.unique(space)).issubset({-1, 0}):
            raise ValueError("Space must consist of -1s and 0s only.")

        self.space = space
        cells_to_fill = np.sum(~self.space.astype(bool))

        if cells_to_fill % 4 != 0 or not self._check_islands_are_fillable_and_set_smallest_island():
            raise ValueError("Space cannot be filled! Contains at least one island with size not divisible by 4!")

        self._total_blocks_to_place = cells_to_fill // 4
        self._num_blocks_placed = 0
        self._space_updated_callback = space_updated_callback

        if use_rng:
            main_rng = random.Random(rng_seed)

            self._default_start_position = (
                main_rng.randrange(self.space.shape[0]),
                main_rng.randrange(self.space.shape[1]),
            )

            def iterable_type[T](items: Sequence[T]) -> Iterable[T]:
                return RandomOffsetIterable(items=items, seed=main_rng.randrange(2**32))
        else:
            iterable_type = CyclingOffsetIterable
            self._default_start_position = (0, 0)

        self._nested_tetromino_iterable = iterable_type(
            [iterable_type(self._get_unique_rotations_transposes(tetromino)) for tetromino in self.TETROMINOS]
        )
        self._tetromino_idx_neighbor_offset_iterable: Iterable[tuple[int, tuple[int, int]]] = list(
            product(range(4), ((-1, 0), (0, -1), (1, 0), (0, 1)))
        )
        if not top_left_tendency:
            self._tetromino_idx_neighbor_offset_iterable = iterable_type(self._tetromino_idx_neighbor_offset_iterable)

        self._smallest_island: NDArray[np.bool] | None = None
        self._unfillable_cell_position: tuple[int, int] | None = None
        self._finished = False

    @staticmethod
    def _get_unique_rotations_transposes(tetromino: NDArray[np.bool]) -> list[NDArray[np.bool]]:
        """Return a list of all unique rotated or transposed views (not copies!) of the tetromino."""
        unique_tetromino_views: list[NDArray[np.bool]] = []

        for rotations in range(4):
            for transpose in (False, True):
                tetromino_view = np.rot90(tetromino, k=rotations)
                if transpose:
                    tetromino_view = tetromino_view.T

                if not any(np.array_equal(tetromino_view, view) for view in unique_tetromino_views):
                    unique_tetromino_views.append(tetromino_view)

        return unique_tetromino_views

    @property
    def total_block_to_place(self) -> int:
        return self._total_blocks_to_place

    @property
    def num_blocks_placed(self) -> int:
        return self._num_blocks_placed

    def fill(self, start_position: tuple[int, int] | None = None) -> None:
        with ensure_sufficient_recursion_depth(self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN):
            self._fill(cell_to_fill_position=start_position or self._default_start_position)

    def _fill(self, cell_to_fill_position: tuple[int, int] = (0, 0)) -> None:
        if np.all(self.space):
            self._finished = True
            return

        # Prio 1: Try to fill the unfillable cell
        if self._unfillable_cell_position:
            cell_to_fill_position = self._unfillable_cell_position
            self._unfillable_cell_position = None
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

        for next_cell_to_fill_position in self._generate_tetromino_placements(cell_to_fill_position):
            self._fill(cell_to_fill_position=next_cell_to_fill_position)
            if self._finished:
                # when finished, simply unwind the stack all the way to the top, *without* backtracking
                return

        # at this point we either have tried everything to fill `cell_to_fill_position` but were unsuccessful,
        # or are in the process of fast backtracking because we have encountered an unfillable cell deeper down the
        # stack

        # in case we are not already in the process of fast backtracking to a cell we were unable to fill
        if not self._unfillable_cell_position:
            # remember this cell as the cell that could not be filled, then fast backtrack until a cell close to it is
            # removed, hopefully removing the issue that made it unfillable
            self._unfillable_cell_position = cell_to_fill_position

    @staticmethod
    def _closest_position_in_area(allowed_area: NDArray[np.bool], position: tuple[int, int]) -> tuple[int, int]:
        allowed_positions = np.argwhere(allowed_area)
        distances_to_position = np.abs(allowed_positions[:, 0] - position[0]) + np.abs(
            allowed_positions[:, 1] - position[1]
        )
        return tuple(allowed_positions[np.argmin(distances_to_position)])

    def _generate_tetromino_placements(self, cell_to_fill_position: tuple[int, int]) -> Iterator[tuple[int, int]]:
        for tetromino_rotations_iterable in self._nested_tetromino_iterable:
            for tetromino in tetromino_rotations_iterable:
                for cell_position_in_tetromino in np.argwhere(tetromino):
                    tetromino_position = cell_to_fill_position - cell_position_in_tetromino

                    space_view_to_put_tetromino = self.space[
                        tetromino_position[0] : tetromino_position[0] + tetromino.shape[0],
                        tetromino_position[1] : tetromino_position[1] + tetromino.shape[1],
                    ]
                    # In case part of the proposed tetromino placement is partly out of bounds, the slicing syntax above
                    # still works, but the shape of the obtained view is not as expected (height, width).
                    # We use this as an efficient way of checking whether the placement is partly out of bounds:
                    if space_view_to_put_tetromino.shape != tetromino.shape:
                        # proposed tetromino placement is partly out of bounds
                        continue

                    if np.any(np.logical_and(space_view_to_put_tetromino, tetromino)):
                        # proposed tetromino placement overlaps with an already filled cell
                        continue

                    self._num_blocks_placed += 1
                    space_view_to_put_tetromino[tetromino] = self._num_blocks_placed

                    space_view_around_tetromino = self.space[
                        max(tetromino_position[0] - 1, 0) : tetromino_position[0] + tetromino.shape[0] + 1,
                        max(tetromino_position[1] - 1, 0) : tetromino_position[1] + tetromino.shape[1] + 1,
                    ]
                    # For efficiency reasons, we first check only the area directly around the tetromino.
                    # If the empty space in that area is fully connected, then the tetromino placement has not created
                    # a new island which means that the islands_are_fillable check can be skipped. (This is usually the
                    # case.)
                    # Only when the current placement creates new islands of empty space do we have to check that these
                    # islands are all still fillable (have a size divisible by 4).
                    if (
                        self._empty_space_has_multiple_islands(space_view_around_tetromino)
                        and not self._check_islands_are_fillable_and_set_smallest_island()
                    ):
                        # proposed tetromino placement would create at least one island of empty space with a size not
                        # divisible by 4, thus not being fillable by tetrominos
                        self._num_blocks_placed -= 1
                        space_view_to_put_tetromino[tetromino] = 0
                        continue

                    # We have just placed a tetromino in the space.
                    # Allow the user to act based on the current state of the space by calling their callback
                    # (if provided).
                    if self._space_updated_callback is not None:
                        self._space_updated_callback()

                    next_cell_to_fill_position = self._get_neighboring_empty_cell_with_least_empty_neighbors_position(
                        tetromino=tetromino, tetromino_position=tetromino_position
                    )
                    if next_cell_to_fill_position is None:
                        # no neighbors to fill next: yield the cell that was just filled and let _fill find the closest
                        # empty cell to it
                        next_cell_to_fill_position = cell_to_fill_position
                        # in this case we should also make sure that self._smallest_island is set because the next cell
                        # to be filled should be inside the now smallest island (in case there are multiple islands)
                        if self._smallest_island is None:
                            # note that space_fillable sets the value of _smallest_island
                            self._check_islands_are_fillable_and_set_smallest_island()

                    yield next_cell_to_fill_position

                    self._num_blocks_placed -= 1
                    space_view_to_put_tetromino[tetromino] = 0

                    # We  have just removed a tetromino from the space because we are backtracking.
                    # Allow the user to act based on the current state of the space by calling their callback
                    # (if provided).
                    if self._space_updated_callback is not None:
                        self._space_updated_callback()

                    if self._unfillable_cell_position and not self._is_close(
                        tetromino=tetromino,
                        tetromino_position=tetromino_position,
                        cell_position=self._unfillable_cell_position,
                    ):
                        # We assume that if the block we have just removed during backtracking is not close to the
                        # unfillable cell, then this change has likely not made the unfillable block fillable again,
                        # thus we don't even try.
                        # (Note that not returning would lead to the next loop iteration, which would lead to yielding,
                        # which would lead to calling _fill, which would lead to trying to fill the unfillable cell)
                        return

    @staticmethod
    def _empty_space_has_multiple_islands(space: NDArray[np.int32]) -> bool:
        island_map = (space == 0).astype(np.uint8)
        return measure.label(island_map, connectivity=1, return_num=True)[1] > 1

    def _check_islands_are_fillable_and_set_smallest_island(self) -> bool:
        """
        Check if the current state of the space has multiple islands of empty space, and if so, whether all islands
        are all still fillable (have a size divisible by 4).

        If so, set self._smallest_island to the smallest island (boolean array being True at the cells belonging to the
        smallest island).

        Return whether all islands are fillable.
        """
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

    def _get_neighboring_empty_cell_with_least_empty_neighbors_position(
        self, tetromino: NDArray[np.bool], tetromino_position: tuple[int, int]
    ) -> tuple[int, int] | None:
        min_empty_neighbors = 4
        min_empty_neighbors_position: tuple[int, int] | None = None

        tetromino_cell_positions = np.argwhere(tetromino) + tetromino_position
        assert len(tetromino_cell_positions) == 4

        for tetromino_cell_idx, neighbor_offset in self._tetromino_idx_neighbor_offset_iterable:
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

    def _is_close(
        self, tetromino: NDArray[np.bool], tetromino_position: tuple[int, int], cell_position: tuple[int, int]
    ) -> bool:
        tetromino_cell_positions = np.argwhere(tetromino) + tetromino_position
        tetromino_to_cell_distances = np.sum(np.abs(tetromino_cell_positions - cell_position), axis=1)
        min_distance = np.min(tetromino_to_cell_distances)
        return min_distance <= self.CLOSE_DISTANCE_THRESHOLD
