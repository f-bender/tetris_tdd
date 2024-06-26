from enum import Enum, auto
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class BlockType(Enum):
    SQUARE = auto()
    L = auto()
    J = auto()
    T = auto()
    Z = auto()
    S = auto()
    I = auto()  # noqa: E741


class Block:
    def __init__(self, block_type: BlockType) -> None:
        match block_type:
            case BlockType.SQUARE:
                self.cells = np.array(
                    [
                        [1, 1],
                        [1, 1],
                    ],
                    dtype=np.bool,
                )
            case BlockType.L:
                self.cells = np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [1, 0, 0],
                    ],
                    dtype=np.bool,
                )
            case BlockType.J:
                self.cells = np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 1],
                    ],
                    dtype=np.bool,
                )
            case BlockType.T:
                self.cells = np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                    ],
                    dtype=np.bool,
                )
            case BlockType.Z:
                self.cells = np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 0],
                        [0, 1, 1],
                    ],
                    dtype=np.bool,
                )
            case BlockType.S:
                self.cells = np.array(
                    [
                        [0, 0, 0],
                        [0, 1, 1],
                        [1, 1, 0],
                    ],
                    dtype=np.bool,
                )
            case BlockType.I:
                self.cells = np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.bool,
                )
            case _:
                raise ValueError(f"Unknown block type: {block_type}")

        # all blocks should have a square box of cells within which they are rotated (even if there would be a smaller
        # rectangular bounding box for the shape)
        assert self.cells.shape[0] == self.cells.shape[1]

        # the square box of cells should be the smallest possible one that can contain all active cells
        # also, the shape should be connected (not explicitly checked, but assumed in this check)
        assert np.all(np.any(self.cells, axis=1)) or np.all(np.any(self.cells, axis=0))

    @property
    def sidelength(self) -> int:
        return self.cells.shape[0]

    @cached_property
    def actual_bounding_box(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the actual bounding box of the active cells within the block."""
        rows_with_active_cells = np.where(np.any(self.cells, axis=1))[0]
        first_row_with_active_cells, last_row_with_active_cells = rows_with_active_cells[0], rows_with_active_cells[-1]

        cols_with_active_cells = np.where(np.any(self.cells, axis=0))[0]
        first_col_with_active_cells, last_col_with_active_cells = cols_with_active_cells[0], cols_with_active_cells[-1]

        return (
            first_row_with_active_cells,
            first_col_with_active_cells,
        ), (
            last_row_with_active_cells + 1,
            last_col_with_active_cells + 1,
        )

    def _invalidate_actual_bounding_box_cache(self) -> None:
        try:
            del self.actual_bounding_box
        except AttributeError:
            pass

    @property
    def actual_cells(self) -> NDArray[np.bool]:
        """Returns the actual cells of the block."""
        return self.cells[
            self.actual_bounding_box[0][0] : self.actual_bounding_box[1][0],
            self.actual_bounding_box[0][1] : self.actual_bounding_box[1][1],
        ]

    def __str__(self) -> str:
        return "\n".join("".join("X" if cell else "." for cell in row) for row in self.cells)

    def rotate_left(self) -> None:
        self.cells = np.rot90(self.cells)
        self._invalidate_actual_bounding_box_cache()

    def rotate_right(self) -> None:
        self.cells = np.rot90(self.cells, k=-1)
        self._invalidate_actual_bounding_box_cache()
