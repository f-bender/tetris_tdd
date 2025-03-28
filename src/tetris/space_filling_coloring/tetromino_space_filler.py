import contextlib
import inspect
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray
from skimage import measure, segmentation

from tetris.exceptions import BaseTetrisError
from tetris.game_logic.components.block import Block, BlockType
from tetris.space_filling_coloring.offset_iterables import CyclingOffsetIterable, RandomOffsetIterable


class NotFillableError(BaseTetrisError):
    def __init__(self, unfillable_cell_position: tuple[int, int] | None) -> None:
        super().__init__(
            "Space could not be filled! "
            "It likely contains some empty cells in a configuration that are impossible to fill with tetrominos. "
            f"Unfillable cell position: {unfillable_cell_position}"
        )


class TetrominoSpaceFiller:
    TETROMINOS = tuple(
        Block(block_type).actual_cells.astype(bool)
        for block_type in (
            # J and Z are omitted since transposing, i.e. mirroring is done in the process of filling
            BlockType.T,
            BlockType.O,
            BlockType.I,
            BlockType.L,
            BlockType.S,
        )
    )
    STACK_FRAMES_SAFETY_MARGIN = 10
    CLOSE_DISTANCE_THRESHOLD = 5
    TETROMINO_SIZE = 4

    def __init__(
        self,
        space: NDArray[np.int32],
        *,
        use_rng: bool = True,
        rng_seed: int | None = None,
        top_left_tendency: bool = True,
        space_updated_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled (inplace). Zeros will be filled in by tetrominos. -1s are considered holes and
                will be left as is.
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
            msg = "rng_seed should only be set when use_rng is True"
            raise ValueError(msg)

        values = set(np.unique(space))
        if not values <= {-1, 0}:
            msg = "Space must consist of -1s and 0s only."
            raise ValueError(msg)

        if 0 not in values:
            msg = "No zeros in space; there is nothing to be filled!"
            raise ValueError(msg)

        # 0: space to be filled, -1: holes not to be filled
        self.space = space

        self._smallest_island: NDArray[np.bool] | None = None
        self._unfillable_cell_position: tuple[int, int] | None = None
        self._last_unfillable_cell_position: tuple[int, int] | None = None
        self._unfillable_cell_neighborhood: NDArray[np.bool] | None = None
        self._finished = False

        cells_to_fill = np.sum(~self.space.astype(bool))

        if cells_to_fill % self.TETROMINO_SIZE != 0 or not self._check_islands_are_fillable_and_set_smallest_island():
            msg = (
                "Space cannot be filled! "
                f"Contains at least one island with size not divisible by {self.TETROMINO_SIZE}!"
            )
            raise ValueError(msg)

        # 0: space to be filled, >0: holes, each hole with an individual value, starting at 1
        self._space_with_labeled_holes: NDArray[np.int64] = measure.label(
            (self.space == -1).view(np.uint8), connectivity=2
        )

        self._total_blocks_to_place = cells_to_fill // self.TETROMINO_SIZE

        self._num_placed_blocks = 0
        self._space_updated_callback = space_updated_callback

        self._default_start_position = (0, 0)
        if use_rng:
            main_rng = random.Random(rng_seed)

            def iterable_type[T](items: Sequence[T]) -> Iterable[T]:
                return RandomOffsetIterable(items=items, seed=main_rng.randrange(2**32))

            if not top_left_tendency:
                self._default_start_position = (
                    main_rng.randrange(self.space.shape[0]),
                    main_rng.randrange(self.space.shape[1]),
                )
        else:
            iterable_type = CyclingOffsetIterable

        self._nested_tetromino_iterable = iterable_type(
            [iterable_type(self._get_unique_rotations_transposes(tetromino)) for tetromino in self.TETROMINOS],
        )
        self._tetromino_idx_neighbor_offset_iterable: Iterable[tuple[int, tuple[int, int]]] = list(
            product(range(self.TETROMINO_SIZE), ((-1, 0), (0, -1), (1, 0), (0, 1))),
        )
        # if we iterate over self._tetromino_idx_neighbor_offset_iterable as is, top and left neighbors will always be
        # checked first, and if they have 3 neighbors, they are early returned without ever checking more neighbors,
        # thus establishing a tendency for the "movement" of the filling algorithm towards the top left
        if not top_left_tendency:
            # if this is not desired, use the offset iterable type to use different offsets into the list when iterating
            self._tetromino_idx_neighbor_offset_iterable = iterable_type(self._tetromino_idx_neighbor_offset_iterable)

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
    def total_blocks_to_place(self) -> int:
        return self._total_blocks_to_place

    @property
    def num_placed_blocks(self) -> int:
        return self._num_placed_blocks

    @property
    def finished(self) -> bool:
        return self._finished

    @contextlib.contextmanager
    def _ensure_sufficient_recursion_depth(self) -> Iterator[None]:
        required_depth = len(inspect.stack()) + self._total_blocks_to_place + self.STACK_FRAMES_SAFETY_MARGIN

        prev_depth = sys.getrecursionlimit()
        sys.setrecursionlimit(max(required_depth, prev_depth))
        try:
            yield
        finally:
            sys.setrecursionlimit(prev_depth)

    def fill(self, start_position: tuple[int, int] | None = None) -> None:
        """Fill up the space with tetrominos, each identified by a unique number, inplace."""
        with self._ensure_sufficient_recursion_depth():
            self._fill(cell_to_fill_position=start_position or self._default_start_position)

        if not self._finished:
            raise NotFillableError(self._unfillable_cell_position)

    def _fill(self, cell_to_fill_position: tuple[int, int]) -> None:
        for next_cell_to_fill_position in self._generate_tetromino_placements(cell_to_fill_position):
            self._fill(cell_to_fill_position=next_cell_to_fill_position)
            if self._finished:
                # when finished, simply unwind the stack all the way to the top, *without* backtracking
                return

    def ifill(self, start_position: tuple[int, int] | None = None) -> Iterator[None]:
        """Iterator version of fill(). After every update of the space, control is yielded to the caller."""
        with self._ensure_sufficient_recursion_depth():
            yield from self._ifill(cell_to_fill_position=start_position or self._default_start_position)

        if not self._finished:
            raise NotFillableError(self._unfillable_cell_position)

    def _ifill(self, cell_to_fill_position: tuple[int, int]) -> Iterator[None]:
        for next_cell_to_fill_position in self._generate_tetromino_placements(cell_to_fill_position):
            # We have just placed a tetromino in the space.
            # Allow the caller of this iterator to act based on the current state of the space before handing back
            # control to us through the next call of "next()".
            yield

            yield from self._ifill(cell_to_fill_position=next_cell_to_fill_position)
            if self._finished:
                return

            # We (probably) have just removed a tetromino from the space because we are backtracking.
            # Allow the caller of this iterator to act based on the current state of the space before handing back
            # control to us through the next call of "next()".
            yield

    def _updated_cell_to_fill_position(self, cell_to_fill_position: tuple[int, int]) -> tuple[int, int]:
        try:
            # Prio 1: Try to fill the unfillable cell if there is one
            if self._unfillable_cell_position:
                try:
                    if self.space[self._unfillable_cell_position] == 0:
                        # only if unfillable cell is actually still unfilled, set it as the next position to fill
                        return self._unfillable_cell_position
                finally:
                    # unset unfillable cell regardless: if we didn't returned, it is already filled, otherwise it will
                    # be in this step
                    self._unfillable_cell_position = None

            # Prio 2: Fill the smallest island (in case the requested position is not inside it, change it)
            if self._smallest_island is not None and not self._smallest_island[cell_to_fill_position]:
                return self._closest_position_in_area(
                    allowed_area=self._smallest_island, position=cell_to_fill_position
                )

            # Prio 3: Select an empty cell as close as possible to the requested cell
            # (the cell itself in case it's empty)
            if self.space[cell_to_fill_position] != 0:
                return self._closest_position_in_area(allowed_area=self.space == 0, position=cell_to_fill_position)

            return cell_to_fill_position
        finally:
            # whether we used it or not, unset self._smallest_island as it will be invalid after the next placement
            self._smallest_island = None

    @staticmethod
    def _closest_position_in_area(allowed_area: NDArray[np.bool], position: tuple[int, int]) -> tuple[int, int]:
        allowed_positions = np.argwhere(allowed_area)
        distances_to_position = np.abs(allowed_positions[:, 0] - position[0]) + np.abs(
            allowed_positions[:, 1] - position[1],
        )
        return tuple(allowed_positions[np.argmin(distances_to_position)])

    def _generate_tetromino_placements(  # noqa: C901, PLR0912
        self,
        cell_to_fill_position: tuple[int, int],
    ) -> Iterator[tuple[int, int]]:
        cell_to_fill_position = self._updated_cell_to_fill_position(cell_to_fill_position)

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

                    self._num_placed_blocks += 1
                    space_view_to_put_tetromino[tetromino] = self._num_placed_blocks

                    space_view_around_tetromino = self.space[
                        max(tetromino_position[0] - 1, 0) : tetromino_position[0] + tetromino.shape[0] + 1,
                        max(tetromino_position[1] - 1, 0) : tetromino_position[1] + tetromino.shape[1] + 1,
                    ]
                    # For efficiency reasons, we first check only the area directly around the tetromino.
                    # If the empty space in that area is fully connected, then the tetromino placement has not created
                    # a new island which means that the islands_are_fillable check can be skipped. (This is usually the
                    # case.)
                    # Only when the current placement creates new islands of empty space do we have to check that these
                    # islands are all still fillable (have a size divisible by TETROMINO_SIZE (4)).
                    if (
                        self._empty_space_has_multiple_islands(space_view_around_tetromino)
                        and not self._check_islands_are_fillable_and_set_smallest_island()
                    ):
                        # Proposed tetromino placement would create at least one island of empty space with a size not
                        # divisible by TETROMINO_SIZE (4), thus not being fillable by tetrominos
                        # So we cancel the placement and skip to the next loop iteration
                        self._num_placed_blocks -= 1
                        space_view_to_put_tetromino[tetromino] = 0
                        continue

                    # We have just placed a tetromino in the space.
                    # Allow the user to act based on the current state of the space by calling their callback
                    # (if provided).
                    if self._space_updated_callback is not None:
                        self._space_updated_callback()

                    if self._num_placed_blocks == self._total_blocks_to_place:
                        # we are done and trigger the immediate unwinding of the stack without backtracking
                        self._finished = True
                        return

                    next_cell_to_fill_position = self._get_neighboring_empty_cell_with_least_empty_neighbors_position(
                        tetromino=tetromino,
                        tetromino_position=tetromino_position,
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

                    self._num_placed_blocks -= 1
                    space_view_to_put_tetromino[tetromino] = 0

                    # We  have just removed a tetromino from the space because we are backtracking.
                    # Allow the user to act based on the current state of the space by calling their callback
                    # (if provided).
                    if self._space_updated_callback is not None:
                        self._space_updated_callback()

                    if (
                        self._unfillable_cell_position
                        and not self._tetromino_overlaps_with_unfillable_cell_neighborhood(
                            tetromino=tetromino, tetromino_position=tetromino_position
                        )
                        # hail mary: if we are at the very top stack frame, about to fail the algorithm, still try if
                        # our remaining options here make it work
                    ) and self._num_placed_blocks != 0:
                        # We assume that if the block we have just removed during backtracking is not close to the
                        # unfillable cell, then this change has likely not made the unfillable block fillable again,
                        # thus we don't even try.
                        # (Note that not returning would lead to the next loop iteration, which would lead to yielding,
                        # which would lead to calling _fill, which would lead to trying to fill the unfillable cell)
                        return

        # at this point we either have tried everything to fill `cell_to_fill_position` but were unsuccessful,
        # or are in the process of fast backtracking because we have encountered an unfillable cell deeper down the
        # stack

        # in case we are not already in the process of fast backtracking to a cell we were unable to fill
        if not self._unfillable_cell_position:
            # remember this cell as the cell that could not be filled, then fast backtrack until a cell close to it is
            # removed, hopefully removing the issue that made it unfillable
            self._unfillable_cell_position = cell_to_fill_position

            # in case the last unfillable cell was the same as this, we can re-use the unfillable_cell_neighborhood
            if self._unfillable_cell_position != self._last_unfillable_cell_position:
                self._unfillable_cell_neighborhood = self._get_neighborhood(cell_to_fill_position)

            self._last_unfillable_cell_position = self._unfillable_cell_position

    def _tetromino_overlaps_with_unfillable_cell_neighborhood(
        self, tetromino: NDArray[np.bool], tetromino_position: tuple[int, int]
    ) -> bool:
        assert self._unfillable_cell_neighborhood is not None
        return bool(
            np.any(
                self._unfillable_cell_neighborhood[tetromino_position[0] :, tetromino_position[1] :][
                    np.where(tetromino)
                ]
            )
        )

    def _get_neighborhood(self, position: tuple[int, int]) -> NDArray[np.bool]:
        distances: NDArray[np.int64] = np.sum(
            np.abs(np.indices(self.space.shape) - np.array(position)[:, np.newaxis, np.newaxis]), axis=0
        )
        # cells with a manhattan distance less than `CLOSE_DISTANCE_THRESHOLD` are considered to be in the neighborhood
        neighborhood: NDArray[np.bool] = distances <= self.CLOSE_DISTANCE_THRESHOLD

        # also, if holes (areas that shall not be filled) are within the neighborhood, their entire boundary is by
        # extension also considered to be in the neighborhood
        if np.any(hole_labels := self._space_with_labeled_holes[neighborhood]):
            # remove 0, which is just background (not a hole)
            hole_labels = [label for label in np.unique(hole_labels) if label != 0]

            space_with_nearby_holes: NDArray[np.bool] = np.isin(self._space_with_labeled_holes, hole_labels)
            holes_boundary: NDArray[np.bool] = segmentation.find_boundaries(
                space_with_nearby_holes, mode="outer", connectivity=2
            )
            neighborhood[holes_boundary] = True

        return neighborhood

    @staticmethod
    def _empty_space_has_multiple_islands(space: NDArray[np.int32]) -> bool:
        island_map = (space == 0).view(np.uint8)
        return measure.label(island_map, connectivity=1, return_num=True)[1] > 1

    def _check_islands_are_fillable_and_set_smallest_island(self) -> bool:
        """Check that self.space is in a valid state, and set the self._smallest_island attribute.

        Specifically, check if the current state of the space has multiple islands of empty space, and if so, whether
        all islands are all still fillable (have a size divisible by TETROMINO_SIZE (4)).
        If so, set self._smallest_island to the smallest island (boolean array being True at the cells belonging to the
        smallest island).

        Return a bool whether the space is in a valid state (all islands are fillable).
        """
        island_map = (self.space == 0).view(np.uint8)
        space_with_labeled_islands, num_islands = measure.label(island_map, connectivity=1, return_num=True)

        if num_islands <= 1:
            self._smallest_island = None
            return True

        smallest_island_size = self.space.shape[0] * self.space.shape[1]

        for i in range(1, num_islands + 1):
            island = space_with_labeled_islands == i
            island_size = np.sum(island)

            if island_size % self.TETROMINO_SIZE != 0:
                self._smallest_island = None
                return False

            if island_size < smallest_island_size:
                smallest_island_size = island_size
                self._smallest_island = island

        return True

    def _get_neighboring_empty_cell_with_least_empty_neighbors_position(
        self,
        tetromino: NDArray[np.bool],
        tetromino_position: tuple[int, int],
    ) -> tuple[int, int] | None:
        min_empty_neighbors = 4
        min_empty_neighbors_position: tuple[int, int] | None = None

        tetromino_cell_positions = np.argwhere(tetromino) + tetromino_position
        assert len(tetromino_cell_positions) == self.TETROMINO_SIZE

        for tetromino_cell_idx, neighbor_offset in self._tetromino_idx_neighbor_offset_iterable:
            neighbor_position = tetromino_cell_positions[tetromino_cell_idx] + neighbor_offset

            if (
                not 0 <= neighbor_position[0] < self.space.shape[0]
                or not 0 <= neighbor_position[1] < self.space.shape[1]
                or self.space[*neighbor_position] != 0
            ):
                # neighbor is out of bounds, or already filled
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
    def space_can_be_filled(space: NDArray[np.int32]) -> bool:
        """Check that space is (probably) fillable, i.e. all its islands have a size divisible by TETROMINO_SIZE (4)."""
        if not np.any(space == 0):
            # if there are no cells to be filled at all, we consider the space not fillable
            return False

        island_map = (space == 0).view(np.uint8)
        space_with_labeled_islands, num_islands = measure.label(island_map, connectivity=1, return_num=True)
        return all(
            np.sum(space_with_labeled_islands == i) % TetrominoSpaceFiller.TETROMINO_SIZE == 0
            for i in range(1, num_islands + 1)
        )

    @staticmethod
    def validate_filled_space(filled_space: NDArray[np.int32]) -> None:
        """Validate that the filled space is a valid tetromino space filling.

        Raises:
            ValueError: If the filled space is invalid.
        """
        if np.any(filled_space == 0):
            msg = "There are empty spaces left, space has not been completely filled."
            raise ValueError(msg)

        tetromino_idxs = [idx for idx in np.unique(filled_space) if idx > 0]
        if sorted(tetromino_idxs) != list(range(1, np.sum(filled_space > 0) // 4 + 1)):
            msg = "Tetromino indices are not consecutive."
            raise ValueError(msg)

        for idx in tetromino_idxs:
            # check that each tetromino is a single connected component of size 4
            island_map = (filled_space == idx).view(np.uint8)
            if np.sum(island_map) != TetrominoSpaceFiller.TETROMINO_SIZE:
                msg = f"Tetromino {idx} isn't of size 4."
                raise ValueError(msg)

            if measure.label(island_map, connectivity=1, return_num=True)[1] != 1:
                msg = f"Tetromino {idx} is not a single connected component."
                raise ValueError(msg)
