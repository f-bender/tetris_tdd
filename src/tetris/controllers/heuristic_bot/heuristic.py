"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.board import Board


class Heuristic(NamedTuple):
    """A Heuristic for evaluating a board state."""

    # mypy doesn't seem to understand that this is a class variables
    CRITICAL_GAP_HEIGHT = 3  # type: ignore[misc]

    # these parameters reached 32,779 cleared lines before dying (in genetic algorithm)
    sum_of_cell_heights_close_to_top_weight: float = 0.2194666215297296
    num_distinct_overhangs_weight: float = 3.8655963791975814
    num_rows_with_overhung_holes_weight: float = 4.842373676164849
    num_overhanging_and_overhung_cells_weight: float = 0.10762127391171512
    num_narrow_gaps_weight: float = 3.243331719348859
    sum_of_cell_heights_weight: float = 0.055269909402115645
    sum_of_adjacent_height_differences_weight: float = 1.4710964800855681

    close_to_top_threshold: int = 1

    def loss(self, board: Board) -> float:
        """Compute a measure of how bad a board is."""
        board_array = board.array_view_without_active_block().astype(bool)

        adjacent_height_differences = self.adjacent_height_differences(board_array)

        # fmt: off
        return (
            + (self.sum_of_cell_heights_close_to_top_weight   * self.sum_cell_heights_close_to_top(board_array, self.close_to_top_threshold))  # noqa: E501
            + (self.num_distinct_overhangs_weight             * self.count_rows_with_overhung_cells(board_array))
            + (self.num_rows_with_overhung_holes_weight       * self.count_distinct_overhangs(board_array))
            + (self.num_overhanging_and_overhung_cells_weight * self.count_overhanging_and_overhung_cells(board_array))
            + (self.num_narrow_gaps_weight                    * self.count_narrow_gaps(adjacent_height_differences))
            + (self.sum_of_cell_heights_weight                * self.sum_cell_heights(board_array))
            + (self.sum_of_adjacent_height_differences_weight * np.sum(np.abs(adjacent_height_differences)))
        )
        # fmt: on

    @staticmethod
    def sum_cell_heights_close_to_top(board_array: NDArray[np.bool], close_threshold: int) -> int:
        # higher cells should be weighted higher - sum_cell_heights fits perfectly
        return Heuristic.sum_cell_heights(board_array[:close_threshold])

    @staticmethod
    def count_rows_with_overhung_cells(board_array: NDArray[np.bool]) -> int:
        return len(
            {
                empty_cell_idx
                for column in np.rot90(board_array, k=1)
                if np.any(column)
                for empty_cell_idx in np.where(~column)[0]
                if empty_cell_idx > np.argmax(column)
            }
        )

    @staticmethod
    def count_distinct_overhangs(board_array: NDArray[np.bool]) -> int:
        return np.sum(np.diff(board_array.astype(np.int8), axis=0) == -1)

    @staticmethod
    def count_overhanging_and_overhung_cells(board_array: NDArray[np.bool]) -> int:
        return sum(
            # overhaning active cells
            np.sum(column[np.argmin(column) :])
            for column in np.rot90(board_array, k=-1)
            if not np.all(column)
        ) + sum(
            # overhung empty cells
            np.sum(~column[np.argmax(column) :])
            for column in np.rot90(board_array, k=1)
            if np.any(column)
        )

    @staticmethod
    def count_narrow_gaps(adjacent_height_differences: NDArray[np.int_]) -> int:
        """Number of gaps that can only be filled by I pieces."""
        return (
            np.sum(
                np.logical_and(
                    adjacent_height_differences[:-1] <= -Heuristic.CRITICAL_GAP_HEIGHT,
                    adjacent_height_differences[1:] >= Heuristic.CRITICAL_GAP_HEIGHT,
                )
            )
            + int(adjacent_height_differences[0] >= Heuristic.CRITICAL_GAP_HEIGHT)
            + int(adjacent_height_differences[-1] <= -Heuristic.CRITICAL_GAP_HEIGHT)
        )

    @staticmethod
    def sum_cell_heights(board_array: NDArray[np.bool]) -> int:
        # get the y indices of all active cells and compute their sum
        # the result is the summed distance of the active cells to the top of the board, so subtract the result from the
        # board height multiplied by the number of active cells to get the summed cell heights counting from the bottom
        y_indices = np.nonzero(board_array)[0]
        return int(y_indices.size * board_array.shape[0] - np.sum(y_indices))

    @staticmethod
    def adjacent_height_differences(board_array: NDArray[np.bool]) -> NDArray[np.int_]:
        # get the highest active cell index for each column, compute adjacent differences
        # in case there are no active cells in a column, use the full height of the array as the height (i.e. consider
        # the highest active cell to be below the bottom)
        return -np.diff(
            np.where(
                np.any(board_array, axis=0),
                np.argmax(board_array, axis=0),
                board_array.shape[0],
            )
        )


def mutated_heuristic(heuristic: Heuristic, mutation_rate: float = 1.0) -> Heuristic:
    rng = np.random.default_rng()

    return Heuristic(  # type: ignore[call-arg]
        sum_of_cell_heights_close_to_top_weight=_mutate_weight(
            weight=heuristic.sum_of_cell_heights_close_to_top_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        num_distinct_overhangs_weight=_mutate_weight(
            weight=heuristic.num_distinct_overhangs_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        num_rows_with_overhung_holes_weight=_mutate_weight(
            weight=heuristic.num_rows_with_overhung_holes_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        num_overhanging_and_overhung_cells_weight=_mutate_weight(
            weight=heuristic.num_overhanging_and_overhung_cells_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        num_narrow_gaps_weight=_mutate_weight(
            weight=heuristic.num_narrow_gaps_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        sum_of_cell_heights_weight=_mutate_weight(
            weight=heuristic.sum_of_cell_heights_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        sum_of_adjacent_height_differences_weight=_mutate_weight(
            weight=heuristic.sum_of_adjacent_height_differences_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        close_to_top_threshold=round(
            np.clip(
                heuristic.close_to_top_threshold * rng.normal(1, mutation_rate),
                1,
                19,
            )
        ),
    )


def _mutate_weight(
    weight: float, mutation_rate: float, rng: np.random.Generator, clip_range: tuple[float, float] = (1e-10, 1e10)
) -> float:
    return float(
        np.clip(
            weight * (1 + mutation_rate) ** rng.standard_normal(),
            *clip_range,
        )
    )
