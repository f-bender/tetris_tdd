from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components import Block
from tetris.game_logic.components.block import BoundingBox, Vec
from tetris.game_logic.components.exceptions import (
    ActiveBlockOverlapError,
    CannotDropBlockError,
    CannotNudgeError,
    CannotSpawnBlockError,
    NoActiveBlockError,
)


@dataclass(frozen=True, slots=True)
class PositionedBlock:
    block: Block
    position: Vec

    @property
    def actual_bounding_box(self) -> BoundingBox:
        return self.block.actual_bounding_box + self.position


class Board:
    def __init__(self) -> None:
        # initialized in a degenerate state - don't call constructor but rather one of the creation classmethods
        self._board: NDArray[np.uint8] = np.zeros((0, 0), dtype=np.uint8)
        self.active_block: PositionedBlock | None = None

    @classmethod
    def create_empty(cls, height: int, width: int) -> Self:
        board = cls()
        board._board = np.zeros((height, width), dtype=np.uint8)
        return board

    @classmethod
    def from_string_representation(cls, string: str) -> Self:
        if not set(string) <= {"X", ".", " ", "\n"}:
            msg = (
                "Invalid string representation of board! "
                f"Must consist of only 'X', '.', spaces, and newlines, but found {set(string)}"
            )
            raise ValueError(msg)

        width = 0
        _board: list[list[bool]] = []
        for line in string.strip().splitlines():
            line = line.strip()  # noqa: PLW2901
            if not width:
                width = len(line)
            elif len(line) != width:
                msg = "Invalid string representation of board (all lines must have the same width)"
                raise ValueError(msg)

            _board.append([c == "X" for c in line])

        board = cls()
        board._board = np.array(_board, dtype=np.uint8)
        return board

    def set_from_other(self, other: Self) -> None:
        if self._board.shape == other._board.shape:  # noqa: SLF001
            # if the shapes match, we can avoid new memory allocation and just fill the existing array
            self._board[...] = other._board  # noqa: SLF001
        else:
            self._board = other._board.copy()  # noqa: SLF001

        self.active_block = other.active_block

    def as_array(self) -> NDArray[np.uint8]:
        return self._board_with_block()

    def array_view_without_active_block(self) -> NDArray[np.uint8]:
        return self._board

    def set_from_array(
        self, array: NDArray[np.uint8], active_block_displacement: tuple[int, int] | None = None
    ) -> None:
        new_board = array.copy()
        if new_board.shape != self._board.shape:
            msg = "Array shape does not match board shape"
            raise ValueError(msg)

        if self.active_block is not None:
            active_block = (
                PositionedBlock(self.active_block.block, self.active_block.position + Vec(*active_block_displacement))
                if active_block_displacement
                else self.active_block
            )
            if self.positioned_block_overlaps_with_active_cells(active_block, new_board):
                raise ActiveBlockOverlapError

            self.active_block = active_block

        self._board = new_board

    @property
    def height(self) -> int:
        return self._board.shape[0]

    @property
    def width(self) -> int:
        return self._board.shape[1]

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self._board.shape)

    def __str__(self) -> str:
        return "\n".join("".join(("X" if c else ".") for c in line) for line in self._board_with_block())

    def has_active_block(self) -> bool:
        return self.active_block is not None

    def clear(self) -> None:
        self._board = np.zeros_like(self._board, dtype=np.uint8)

    def spawn_random_block(self) -> None:
        self.spawn(Block.create_random())

    def spawn(self, block: Block, position: Vec | None = None) -> None:
        position = position or self._top_middle_position(block)
        self.active_block = PositionedBlock(block, position)

        if not self._positioned_block_is_in_valid_position(self.active_block):
            try:
                self._nudge_active_block_into_valid_position()
            except CannotNudgeError as e:
                self.active_block = None
                raise CannotSpawnBlockError from e

    def try_move_active_block_left(self) -> None:
        self._move(-1)

    def try_move_active_block_right(self) -> None:
        self._move(1)

    def try_rotate_active_block_left(self) -> None:
        self._rotate("left")

    def try_rotate_active_block_right(self) -> None:
        self._rotate("right")

    def drop_active_block(self) -> None:
        if self.active_block is None:
            raise NoActiveBlockError

        dropped_block = PositionedBlock(self.active_block.block, self.active_block.position + Vec(1, 0))

        if not self._bbox_in_bounds(dropped_block.actual_bounding_box):
            msg = "Active block reached bottom of the board"
            raise CannotDropBlockError(msg)

        if self.positioned_block_overlaps_with_active_cells(dropped_block, self._board):
            msg = "Active block has landed on an active cell"
            raise CannotDropBlockError(msg)

        self.active_block = dropped_block

    def merge_active_block(self) -> None:
        if self.active_block is None:
            raise NoActiveBlockError

        self._merge_active_block_into_board(self.active_block, self._board)
        self.active_block = None

    def _top_middle_position(self, block: Block) -> Vec:
        y_offset = block.actual_bounding_box.top_left.y
        return Vec(-y_offset, ceil((self.width - block.sidelength) / 2))

    def _bbox_in_bounds(self, bbox: BoundingBox) -> bool:
        # note: top is not relevant since blocks may stretch beyond the top of the board
        # its spot however left in the signature to avoid confusing arguments
        _, left, bottom, right = bbox
        return not (left < 0 or bottom > self.height or right > self.width)

    def positioned_block_overlaps_with_active_cells(
        self,
        positioned_block: PositionedBlock,
        board: NDArray[np.uint8] | None = None,
    ) -> bool:
        if board is None:
            board = self._board

        top, left, bottom, right = positioned_block.actual_bounding_box

        if bottom <= 0:
            # block is entirely above the top of the board
            return False

        # blocks are allowed to stretch beyond the top line; in this case the corresponding cells are not considered
        # part of the board
        top_cutoff = max(0, -top)
        return bool(
            np.any(
                np.logical_and(
                    board[top + top_cutoff : bottom, left:right], positioned_block.block.actual_cells[top_cutoff:, :]
                )
            )
        )

    def _move(self, x_offset: Literal[-1, 1]) -> None:
        if self.active_block is None:
            raise NoActiveBlockError

        moved_block = PositionedBlock(self.active_block.block, self.active_block.position + Vec(0, x_offset))

        if self._positioned_block_is_in_valid_position(moved_block):
            self.active_block = moved_block

    def _rotate(self, direction: Literal["left", "right"]) -> None:
        if self.active_block is None:
            raise NoActiveBlockError

        if direction == "left":
            self.active_block.block.rotate_left()
        else:
            self.active_block.block.rotate_right()

        if not self._positioned_block_is_in_valid_position(self.active_block):
            try:
                self._nudge_active_block_into_valid_position()
            except CannotNudgeError:
                if direction == "left":
                    self.active_block.block.rotate_right()
                else:
                    self.active_block.block.rotate_left()

    def _positioned_block_is_in_valid_position(self, positioned_block: PositionedBlock) -> bool:
        return self._bbox_in_bounds(
            positioned_block.actual_bounding_box
        ) and not self.positioned_block_overlaps_with_active_cells(positioned_block, self._board)

    def _nudge_active_block_into_valid_position(self) -> None:
        if self.active_block is None:
            raise NoActiveBlockError

        # RULES:
        # we can only nudge laterally (left, right)
        # we can nudge the block at most by half its sidelength
        # we must nudge the least possible amount

        max_nudge = self.active_block.block.sidelength // 2

        for x_offset in sorted(range(-max_nudge, max_nudge + 1), key=abs):
            nudged_block = PositionedBlock(self.active_block.block, self.active_block.position + Vec(0, x_offset))

            if self._positioned_block_is_in_valid_position(nudged_block):
                self.active_block = nudged_block
                return

        raise CannotNudgeError

    def _board_with_block(self) -> NDArray[np.uint8]:
        if self.active_block is not None:
            board = self._board.copy()
            self._merge_active_block_into_board(self.active_block, board)
            return board

        return self._board

    def clear_lines(self, line_idxs: Iterable[int]) -> None:
        for line_idx in line_idxs:
            self.clear_line(line_idx)

    def clear_line(self, line_idx: int) -> None:
        if line_idx < 0 or line_idx >= self.height:
            msg = f"Line index out of range (0-{self.height - 1})"
            raise IndexError(msg)

        # move everything above the cleared line one line down
        self._board[1 : line_idx + 1] = self._board[:line_idx]
        # fill the top line with zeros (empty)
        self._board[0] = np.zeros_like(self._board[0])

    def get_full_line_idxs(self) -> list[int]:
        return list(np.where(np.all(self._board, axis=1))[0])

    @staticmethod
    def _merge_active_block_into_board(positioned_block: PositionedBlock, board: NDArray[np.uint8]) -> None:
        top, left, bottom, right = positioned_block.actual_bounding_box

        if bottom <= 0:
            # block is entirely above the top of the board - doing nothing is silly but this will be a Game Over very
            # soon anyways
            return

        # blocks are allowed to stretch beyond the top line; in this case the corresponding cells are not considered
        # part of the board
        top_cutoff = max(0, -top)

        board[top + top_cutoff : bottom, left:right] |= positioned_block.block.actual_cells[top_cutoff:, :]
