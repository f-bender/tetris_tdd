from dataclasses import dataclass
from math import ceil
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from game_logic.block import Block
from game_logic.exceptions import CannotDropBlock, CannotNudge, CannotSpawnBlock, NoActiveBlock


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
            raise CannotSpawnBlock()

        self._active_block = ActiveBlock(block=block, position=position)

    def _top_middle_position(self, block: Block) -> tuple[int, int]:
        y_offset = block.actual_bounding_box[0][0]
        return -y_offset, ceil((self.width - block.sidelength) / 2)

    def _can_spawn(self, block: Block, position: tuple[int, int]) -> bool:
        (top, left), (bottom, right) = self._actual_block_bounding_box(block, position)

        if (
            not self._bbox_in_bounds(top, left, bottom, right)
            or top < 0  # top of board is considered in bounds, but blocks shouldn't spawn there
            or self._overlaps_with_active_cells(block.actual_cells, top, left, bottom, right)
        ):
            return False

        return True

    def _bbox_in_bounds(self, _: int, left: int, bottom: int, right: int) -> bool:
        # note: top is not relevant since blocks may stretch beyond the top of the board
        # its spot however left in the signature to avoid confusing arguments
        return not (left < 0 or bottom > self.height or right > self.width)

    def _overlaps_with_active_cells(
        self, cells: NDArray[np.bool], top: int, left: int, bottom: int, right: int
    ) -> bool:
        # blocks are allowed to stretch beyond the top line; in this case the corresponding cells are not considered
        # part of the board
        top_cutoff = max(0, -top)
        return bool(np.any(self._board[top + top_cutoff : bottom, left:right] & cells[top_cutoff:, :]))

    def try_move_active_block_left(self) -> None:
        self._move(-1)

    def try_move_active_block_right(self) -> None:
        self._move(1)

    def _move(self, x_offset: Literal[-1, 1]) -> None:
        if self._active_block is None:
            raise NoActiveBlock()

        moved_position = (
            self._active_block.position[0],
            self._active_block.position[1] + x_offset,
        )

        if self._active_block_is_in_valid_position(ActiveBlock(self._active_block.block, position=moved_position)):
            self._active_block.position = moved_position

    def try_rotate_active_block_left(self) -> None:
        self._rotate("left")

    def try_rotate_active_block_right(self) -> None:
        self._rotate("right")

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

        return self._bbox_in_bounds(top, left, bottom, right) and not self._overlaps_with_active_cells(
            active_block.block.actual_cells, top, left, bottom, right
        )

    # TODO: this is more game logic than board scorekeeping -> different responsibility, move into own class, maybe make
    # part of eventual Game class?
    def _nudge_block_into_valid_position(self, active_block: ActiveBlock) -> None:
        # we can only nudge laterally (left, right)
        # we can nudge at most nudge the block by half its sidelength
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

        if self._overlaps_with_active_cells(
            self._active_block.block.actual_cells, dropped_top, dropped_left, dropped_bottom, dropped_right
        ):
            raise CannotDropBlock("Active block has landed on an active cell")

        self._active_block.position = dropped_position

    def __str__(self) -> str:
        return "\n".join("".join(("X" if c else ".") for c in row) for row in self._board_with_block())

    def _board_with_block(self) -> NDArray[np.bool]:
        if self._active_block is not None:
            board = self._board.copy()
            self._merge_active_block_into_board(self._active_block, board)
            return board

        return self._board

    def merge_active_block(self) -> int:
        """Merge the active block into the board.

        If this fills up lines, they are cleared.
        The number of cleared lines is returned.
        """
        if self._active_block is None:
            raise NoActiveBlock()

        self._merge_active_block_into_board(self._active_block, self._board)
        self._active_block = None

        full_line_positions = self._get_full_line_positions_ordered_top_to_bottom()
        self._clear_lines(full_line_positions)
        return len(full_line_positions)

    def _clear_lines(self, full_line_positions: NDArray[np.int_]) -> None:
        for line_position in full_line_positions:
            # move everything above the cleared line position one line down
            self._board[1 : line_position + 1] = self._board[:line_position]
            # fill the top row with zeros (empty)
            self._board[0] = np.zeros_like(self._board[0])

    def _get_full_line_positions_ordered_top_to_bottom(self) -> NDArray[np.int_]:
        return np.where(np.all(self._board, axis=1))[0]

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

    @property
    def height(self) -> int:
        return self._board.shape[0]

    @property
    def width(self) -> int:
        return self._board.shape[1]
