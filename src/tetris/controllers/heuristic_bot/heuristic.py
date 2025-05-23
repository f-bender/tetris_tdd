"""An automated controller that uses a heuristic approach to try and fit the current block into the optimal position."""

import ast
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.board import Board


@dataclass(frozen=True, kw_only=True, slots=True)
class Heuristic:
    """A Heuristic for evaluating a board state."""

    # These default parameters have been obtained through a genetic algorithm optimization
    # and are the (so far) best ones according to a detailed evaluation.

    # On a 20x10 board, evaluated on 200 different seeds:
    # mean score:   49273.64
    # median score: 36714.0
    # max score:    228223
    # min score:    134
    # An assumption of a constant probability of failure to clear the next line models the behavior well.
    # This probability p is 1/mean_score = 1/49273.64 ~= 2e-05.
    # I.e. with a probability of (1-p)**n, we score above n line clears.
    # For example, with a probability of ~82%, we score above 10,000 line clears.
    # All of this assumes standard NES Tetris rules (fully random block spawns (no "random bag"), no "Hold piece").
    sum_of_cell_heights_close_to_top_weight: float = 1_000_000.0
    num_distinct_overhangs_weight: float = 14.28200459973235
    num_rows_with_overhung_holes_weight: float = 5.992567087282148
    num_overhung_cells_weight: float = 0.47296546559291547
    num_overhanging_cells_weight: float = 0.04961999808069783
    num_narrow_gaps_weight: float = 20.434313335543326
    sum_of_cell_heights_weight: float = 0.04895973843887606
    sum_of_adjacent_height_differences_weight: float = 2.9046525787750297

    close_to_top_threshold: int = 2

    _CRITICAL_GAP_HEIGHT: ClassVar[int] = 3

    @classmethod
    def from_repr(cls, heuristic_repr: str) -> Self:
        """Functionally equivalent to `eval(heuristic_repr)` but without the security risk."""
        if heuristic_repr == f"{cls.__name__}()":
            return cls()

        heuristic_dict_literal = (
            heuristic_repr.replace(f"{cls.__name__}(", "{'").replace(")", "}").replace("=", "':").replace(", ", ", '")
        )
        return cls(**ast.literal_eval(heuristic_dict_literal))

    def loss(self, board: Board) -> float:
        """Compute a measure of how bad a board is."""
        # using view() instead of astype() is an optimization that assumes that the int type of the board is 8 bit wide!
        board_array = board.array_view_without_active_block().view(bool)

        adjacent_height_differences = self.adjacent_height_differences(board_array)

        # fmt: off
        return (
            + (self.sum_of_cell_heights_close_to_top_weight   * self.sum_cell_heights_close_to_top(board_array, self.close_to_top_threshold))  # noqa: E501
            + (self.num_distinct_overhangs_weight             * self.count_rows_with_overhung_cells(board_array))
            + (self.num_rows_with_overhung_holes_weight       * self.count_distinct_overhangs(board_array))
            + (self.num_overhung_cells_weight                 * self.count_overhung_cells(board_array))
            + (self.num_overhanging_cells_weight              * self.count_overhanging_cells(board_array))
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
    def count_overhung_cells(board_array: NDArray[np.bool]) -> int:
        # empty cells that are below at least one active cell
        return sum(np.sum(~column[np.argmax(column) :]) for column in np.rot90(board_array, k=1) if np.any(column))

    @staticmethod
    def count_overhanging_cells(board_array: NDArray[np.bool]) -> int:
        # active cells that are above at least one empty cell
        return sum(np.sum(column[np.argmin(column) :]) for column in np.rot90(board_array, k=-1) if not np.all(column))

    @staticmethod
    def count_narrow_gaps(adjacent_height_differences: NDArray[np.int_]) -> int:
        """Number of gaps that can only be filled by I pieces."""
        return (
            np.sum(
                np.logical_and(
                    adjacent_height_differences[:-1] <= -Heuristic._CRITICAL_GAP_HEIGHT,
                    adjacent_height_differences[1:] >= Heuristic._CRITICAL_GAP_HEIGHT,
                )
            )
            + int(adjacent_height_differences[0] >= Heuristic._CRITICAL_GAP_HEIGHT)
            + int(adjacent_height_differences[-1] <= -Heuristic._CRITICAL_GAP_HEIGHT)
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

    return Heuristic(
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
        num_overhung_cells_weight=_mutate_weight(
            weight=heuristic.num_overhung_cells_weight,
            mutation_rate=mutation_rate,
            rng=rng,
        ),
        num_overhanging_cells_weight=_mutate_weight(
            weight=heuristic.num_overhanging_cells_weight,
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
                heuristic.close_to_top_threshold
                + rng.normal(
                    0,
                    # at least keep a ~2% chance of changing it by 1
                    # note: std=0.25 -> 2 std to reach 0.5 -> 0.5 leads to rounding to an adjacent number (2 std -> ~2%)
                    max(mutation_rate, 0.25),
                ),
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
