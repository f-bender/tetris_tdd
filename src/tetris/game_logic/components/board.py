from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components import Block
from tetris.game_logic.components.exceptions import (
    ActiveBlockOverlap,
    CannotDropBlock,
    CannotNudge,
    CannotSpawnBlock,
    NoActiveBlock,
)


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
    def create_empty(cls, height: int, width: int) -> Self:
        board = cls()
        board._board = np.zeros((height, width), dtype=np.bool)
        return board

    @classmethod
    def from_string_representation(cls, string: str) -> Self:
        if not set(string).issubset({"X", ".", " ", "\n"}):
            raise ValueError(
                "Invalid string representation of board! "
                f"Must consist of only 'X', '.', spaces, and newlines, but found {set(string)}"
            )

        width = 0
        _board: list[list[bool]] = []
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

    def as_array(self) -> NDArray[np.bool]:
        return self._board_with_block()

    def as_array_without_active_block(self) -> NDArray[np.bool]:
        return self._board.copy()

    def set_from_array(self, array: NDArray[np.bool]) -> None:
        new_board = array.astype(np.bool)
        if new_board.shape != self._board.shape:
            raise ValueError("Array shape does not match board shape")

        if self._active_block is not None:
            (top, left), (bottom, right) = self._actual_block_bounding_box(
                self._active_block.block, self._active_block.position
            )

            if self._active_block_overlaps_with_active_cells(
                new_board, self._active_block.block.actual_cells, top, left, bottom, right
            ):
                raise ActiveBlockOverlap()

        self._board = new_board

    @property
    def height(self) -> int:
        return self._board.shape[0]

    @property
    def width(self) -> int:
        return self._board.shape[1]

    def __str__(self) -> str:
        return "\n".join("".join(("X" if c else ".") for c in line) for line in self._board_with_block())

    def has_active_block(self) -> bool:
        return self._active_block is not None

    def clear(self) -> None:
        self._board = np.zeros_like(self._board, dtype=np.bool)

    def spawn_random_block(self) -> None:
        self.spawn(Block.create_random())

    def spawn(self, block: Block, position: tuple[int, int] | None = None) -> None:
        position = position or self._top_middle_position(block)
        active_block = ActiveBlock(block, position)

        if not self._active_block_is_in_valid_position(active_block):
            try:
                self._nudge_block_into_valid_position(active_block)
            except CannotNudge as e:
                raise CannotSpawnBlock() from e

        self._active_block = active_block

    def try_move_active_block_left(self) -> None:
        self._move(-1)

    def try_move_active_block_right(self) -> None:
        self._move(1)

    def try_rotate_active_block_left(self) -> None:
        self._rotate("left")

    def try_rotate_active_block_right(self) -> None:
        self._rotate("right")

    def drop_active_block(self) -> None:
        if self._active_block is None:
            raise NoActiveBlock()

        dropped_position = (
            self._active_block.position[0] + 1,
            self._active_block.position[1],
        )

        (dropped_top, dropped_left), (dropped_bottom, dropped_right) = self._actual_block_bounding_box(
            self._active_block.block, dropped_position
        )

        if not self._bbox_in_bounds(dropped_top, dropped_left, dropped_bottom, dropped_right):
            raise CannotDropBlock("Active block reached bottom of the board")

        if self._active_block_overlaps_with_active_cells(
            self._board, self._active_block.block.actual_cells, dropped_top, dropped_left, dropped_bottom, dropped_right
        ):
            raise CannotDropBlock("Active block has landed on an active cell")

        self._active_block.position = dropped_position

    def merge_active_block(self) -> None:
        if self._active_block is None:
            raise NoActiveBlock()

        self._merge_active_block_into_board(self._active_block, self._board)
        self._active_block = None

    def _top_middle_position(self, block: Block) -> tuple[int, int]:
        y_offset = block.actual_bounding_box[0][0]
        return -y_offset, ceil((self.width - block.sidelength) / 2)

    def _bbox_in_bounds(self, _: int, left: int, bottom: int, right: int) -> bool:
        # note: top is not relevant since blocks may stretch beyond the top of the board
        # its spot however left in the signature to avoid confusing arguments
        return not (left < 0 or bottom > self.height or right > self.width)

    @staticmethod
    def _active_block_overlaps_with_active_cells(
        board: NDArray[np.bool], active_block_cells: NDArray[np.bool], top: int, left: int, bottom: int, right: int
    ) -> bool:
        # blocks are allowed to stretch beyond the top line; in this case the corresponding cells are not considered
        # part of the board
        top_cutoff = max(0, -top)
        return bool(np.any(board[top + top_cutoff : bottom, left:right] & active_block_cells[top_cutoff:, :]))

    def _move(self, x_offset: Literal[-1, 1]) -> None:
        if self._active_block is None:
            raise NoActiveBlock()

        moved_position = (
            self._active_block.position[0],
            self._active_block.position[1] + x_offset,
        )

        if self._active_block_is_in_valid_position(ActiveBlock(self._active_block.block, position=moved_position)):
            self._active_block.position = moved_position

    def _rotate(self, direction: Literal["left", "right"]) -> None:
        if self._active_block is None:
            raise NoActiveBlock()

        if direction == "left":
            self._active_block.block.rotate_left()
        else:
            self._active_block.block.rotate_right()

        if not self._active_block_is_in_valid_position(self._active_block):
            try:
                self._nudge_block_into_valid_position(self._active_block)
            except CannotNudge:
                if direction == "left":
                    self._active_block.block.rotate_right()
                else:
                    self._active_block.block.rotate_left()

    def _active_block_is_in_valid_position(self, active_block: ActiveBlock) -> bool:
        (top, left), (bottom, right) = self._actual_block_bounding_box(active_block.block, active_block.position)

        return self._bbox_in_bounds(top, left, bottom, right) and not self._active_block_overlaps_with_active_cells(
            self._board, active_block.block.actual_cells, top, left, bottom, right
        )

    def _nudge_block_into_valid_position(self, active_block: ActiveBlock) -> None:
        # RULES:
        # we can only nudge laterally (left, right)
        # we can nudge the block at most by half its sidelength
        # we must nudge the least possible amount

        max_nudge = active_block.block.sidelength // 2

        for x_offset in sorted(range(-max_nudge, max_nudge + 1), key=abs):
            nudged_position = (
                active_block.position[0],
                active_block.position[1] + x_offset,
            )

            if self._active_block_is_in_valid_position(ActiveBlock(active_block.block, position=nudged_position)):
                active_block.position = nudged_position
                return

        raise CannotNudge()

    def _board_with_block(self) -> NDArray[np.bool]:
        if self._active_block is not None:
            board = self._board.copy()
            self._merge_active_block_into_board(self._active_block, board)
            return board

        return self._board

    def clear_lines(self, line_idxs: Iterable[int]) -> None:
        for line_idx in line_idxs:
            self.clear_line(line_idx)

    def clear_line(self, line_idx: int) -> None:
        if line_idx < 0 or line_idx >= self.height:
            raise IndexError(f"Line index out of range (0-{self.height - 1})")

        # move everything above the cleared line one line down
        self._board[1 : line_idx + 1] = self._board[:line_idx]
        # fill the top line with zeros (empty)
        self._board[0] = np.zeros_like(self._board[0])

    def get_full_line_idxs(self) -> list[int]:
        return list(np.where(np.all(self._board, axis=1))[0])

    @staticmethod
    def _merge_active_block_into_board(active_block: ActiveBlock, board: NDArray[np.bool]) -> None:
        (top, left), (bottom, right) = Board._actual_block_bounding_box(active_block.block, active_block.position)

        # blocks are allowed to stretch beyond the top line; in this case the corresponding cells are not considered
        # part of the board
        top_cutoff = max(0, -top)

        board[top + top_cutoff : bottom, left:right] |= active_block.block.actual_cells[top_cutoff:, :]

    @staticmethod
    def _actual_block_bounding_box(block: Block, position: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        (y_offset, x_offset), (y_end_offset, x_end_offset) = block.actual_bounding_box
        top = position[0] + y_offset
        left = position[1] + x_offset
        bottom = position[0] + y_end_offset
        right = position[1] + x_end_offset

        return (top, left), (bottom, right)
