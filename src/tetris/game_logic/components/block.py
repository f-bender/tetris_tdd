import contextlib
import random
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Vec:
    y: int
    x: int

    def __add__(self, other: Self) -> "Vec":
        return Vec(y=self.y + other.y, x=self.x + other.x)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    top_left: Vec
    bottom_right: Vec

    def __add__(self, position: Vec) -> "BoundingBox":
        return BoundingBox(
            top_left=self.top_left + position,
            bottom_right=self.bottom_right + position,
        )

    def __iter__(self) -> Iterator[int]:
        """Allow unpacking a bounding box into top, left, bottom, and right values."""
        return iter((self.top_left.y, self.top_left.x, self.bottom_right.y, self.bottom_right.x))


class BlockType(IntEnum):
    T = 1
    O = 2  # noqa: E741
    I = 3  # noqa: E741
    L = 4
    S = 5
    J = 6
    Z = 7


class Block:
    def __init__(self, block_type: BlockType) -> None:
        match block_type:
            case BlockType.T:
                cells = [
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ]
            case BlockType.O:
                cells = [
                    [1, 1],
                    [1, 1],
                ]
            case BlockType.I:
                cells = [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                ]
            case BlockType.L:
                cells = [
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                ]
            case BlockType.S:
                cells = [
                    [0, 0, 0],
                    [0, 1, 1],
                    [1, 1, 0],
                ]
            case BlockType.J:
                cells = [
                    [0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 1],
                ]
            case BlockType.Z:
                cells = [
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 1],
                ]
            case _:
                msg = f"Unknown block type: {block_type}"
                raise ValueError(msg)

        self.cells = (np.array(cells) * block_type).astype(np.uint8)

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
    def actual_bounding_box(self) -> BoundingBox:
        """Returns the actual bounding box of the active cells within the block."""
        rows_with_active_cells = np.where(np.any(self.cells, axis=1))[0]
        first_row_with_active_cells, last_row_with_active_cells = (
            int(rows_with_active_cells[0]),
            int(rows_with_active_cells[-1]),
        )

        cols_with_active_cells = np.where(np.any(self.cells, axis=0))[0]
        first_col_with_active_cells, last_col_with_active_cells = (
            int(cols_with_active_cells[0]),
            int(cols_with_active_cells[-1]),
        )

        return BoundingBox(
            top_left=Vec(first_row_with_active_cells, first_col_with_active_cells),
            bottom_right=Vec(last_row_with_active_cells + 1, last_col_with_active_cells + 1),
        )

    def _invalidate_actual_bounding_box_cache(self) -> None:
        with contextlib.suppress(AttributeError):
            del self.actual_bounding_box

    @property
    def actual_cells(self) -> NDArray[np.uint8]:
        """Returns the actual cells of the block."""
        return self.cells[
            self.actual_bounding_box.top_left.y : self.actual_bounding_box.bottom_right.y,
            self.actual_bounding_box.top_left.x : self.actual_bounding_box.bottom_right.x,
        ]

    def __str__(self) -> str:
        return "\n".join("".join("X" if cell else "." for cell in row) for row in self.cells)

    def rotate_left(self) -> None:
        self.cells = np.rot90(self.cells)
        self._invalidate_actual_bounding_box_cache()

    def rotate_right(self) -> None:
        self.cells = np.rot90(self.cells, k=-1)
        self._invalidate_actual_bounding_box_cache()

    def unique_rotations(self) -> list[Self]:
        """Returns all unique rotations of the block."""
        unique_rotations = [deepcopy(self)]

        for _ in range(3):
            new_rotation = deepcopy(unique_rotations[-1])
            new_rotation.rotate_right()

            if not any(
                np.array_equal(registered_rotation.actual_cells, new_rotation.actual_cells)
                for registered_rotation in unique_rotations
            ):
                unique_rotations.append(new_rotation)

        return unique_rotations

    @classmethod
    def create_random(cls, rng: random.Random | None = None) -> Self:
        return cls((rng or random).choice(list(BlockType)))

    def with_cell_value(self, value: int) -> Self:
        """Returns a copy of the block with all active cells set to the given value."""
        new_block = deepcopy(self)
        new_block.cells[new_block.cells != 0] = value
        return new_block
