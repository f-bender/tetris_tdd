from dataclasses import dataclass
from math import ceil
from typing import Self

import numpy as np
from numpy.typing import NDArray

from game_logic.block import Block


@dataclass(slots=True)
class ActiveBlock:
    block: Block
    position: tuple[int, int]


class Board:
    def __init__(self) -> None:
        # initialized in a degenerate state - don't call constructor but rather one of the creation classmethods
        self._board: NDArray[np.bool] = np.zeros((0, 0), dtype=np.bool)
        self._active_block: ActiveBlock | None = None

    @classmethod
    def empty(cls, height: int, width: int) -> Self:
        board = cls()
        board._board = np.zeros((height, width), dtype=np.bool)
        return board

    @classmethod
    def from_string_representation(cls, string: str) -> Self:
        if set(string) != {"X", ".", " ", "\n"}:
            raise ValueError(
                "Invalid string representation of board! "
                f"Must consist of only 'X', '.', spaces, and newlines, but found {set(string)}"
            )

        width = 0
        _board = []
        for line in string.strip().splitlines():
            line = line.strip()
            if not width:
                width = len(line)
            elif len(line) != width:
                raise ValueError("Invalid string representation of board (all lines must have the same width)")

            _board.append([c == "X" for c in line])

        board = cls()
        board._board = np.array(_board, dtype=np.bool)
        return board

    def spawn(self, block: Block, position: tuple[int, int] | None = None) -> None:
        position = position or self._top_middle_position(block)

        if not self._can_spawn(block, position):
            raise ValueError("Cannot spawn block at given position")

        self._active_block = ActiveBlock(block=block, position=position)

    def _top_middle_position(self, block: Block) -> tuple[int, int]:
        y_offset = block.actual_bounding_box[0][0]
        return -y_offset, ceil((self.width - block.sidelength) / 2)

    def has_active_block(self) -> bool:
        return self._active_block is not None

    def _can_spawn(self, block: Block, position: tuple[int, int]) -> bool:
        (top, left), (bottom, right) = self._actual_block_bounding_box(block, position)

        if not self.bbox_fully_in_bounds(top, left, bottom, right) or self.overlaps_with_active_cells(
            block.actual_cells, top, left, bottom, right
        ):
            return False

        return True

    def bbox_fully_in_bounds(self, top, left, bottom, right) -> bool:
        return not (top < 0 or left < 0 or bottom > self.height or right > self.width)

    def overlaps_with_active_cells(self, cells: NDArray[np.bool], top: int, left: int, bottom: int, right: int) -> bool:
        return bool(np.any(self._board[top:bottom, left:right] & cells))

    def __str__(self) -> str:
        return "\n".join("".join(("X" if c else ".") for c in row) for row in self._board_with_block())

    def _board_with_block(self) -> NDArray[np.bool]:
        if self._active_block is not None:
            board = self._board.copy()

            (top, left), (bottom, right) = self._actual_block_bounding_box(
                self._active_block.block, self._active_block.position
            )
            board[top:bottom, left:right] |= self._active_block.block.actual_cells

            return board

        return self._board

    def _actual_block_bounding_box(
        self, block: Block, position: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        (y_offset, x_offset), (y_end_offset, x_end_offset) = block.actual_bounding_box
        top = position[0] + y_offset
        left = position[1] + x_offset
        bottom = position[0] + y_end_offset
        right = position[1] + x_end_offset

        return (top, left), (bottom, right)

    @property
    def height(self) -> int:
        return self._board.shape[0]

    @property
    def width(self) -> int:
        return self._board.shape[1]
