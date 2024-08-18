from collections.abc import Iterator
import time
from typing import NamedTuple

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from game_logic.components.block import Block, BlockType


class PositionedTetromino(NamedTuple):
    position: tuple[int, int]
    tetromino: NDArray[np.bool]


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

    def __init__(self, space: NDArray[np.int16]) -> None:
        """Initialize the space filler.

        Args:
            space: The space to be filled. Zeros will be filled in by tetrominos. -1s are considered holes and will be
                left as is.
        """
        if not set(np.unique(space)).issubset({-1, 0}):
            raise ValueError("Space must consist of -1s and 0s only.")

        if not self.space_fillable(space):
            raise ValueError("Space cannot be filled! Contains at least one island with size not divisible by 4!")

        self.space = space

        self.space_rotated_transposed_views = (
            self.space,
            np.rot90(self.space, k=1).T,
            np.rot90(self.space, k=1),
            np.rot90(self.space, k=2).T,
            np.rot90(self.space, k=2),
            np.rot90(self.space, k=3).T,
            np.rot90(self.space, k=3),
            self.space.T,
        )

    def draw(self) -> None:
        print(cursor.goto(1, 1))
        for row in np.where(self.space == 0, "  ", "XX"):
            print("".join(row))

    def fill(self) -> None:
        iteration = 0
        while 0 in self.space:
            self.draw()
            time.sleep(0.025)

            if not self._place(iteration):
                raise RuntimeError("Could not place any tetromino! Algorithm must be flawed!")
            iteration += 1

    def _place(self, iteration: int) -> bool:
        for tetromino_idx in range(len(self.TETROMINOS)):
            tetromino = self.TETROMINOS[(tetromino_idx + iteration) % len(self.TETROMINOS)]
            if self._place_tetromino(tetromino, iteration):
                return True

        return False

    def _place_tetromino(self, tetromino: NDArray[np.bool], iteration: int) -> bool:
        # TODO efficiency: make sure only distinct versions of the tetromino are used (avoid duplicates because of
        # symmetry (rotational or axial))
        for transpose in (False, True):
            for k in range(4):
                if self._place_rotated_tetromino(
                    np.rot90(tetromino, k=k).T if transpose else np.rot90(tetromino, k=k), iteration
                ):
                    return True

        return False

    def _place_rotated_tetromino(self, tetromino: NDArray[np.bool], iteration: int) -> bool:
        for view_idx in range(len(self.space_rotated_transposed_views)):
            space_view = self.space_rotated_transposed_views[
                (view_idx + iteration) % len(self.space_rotated_transposed_views)
            ]
            if self._place_tetromino_on_view(space_view, tetromino, iteration):
                return True

        return False

    def _place_tetromino_on_view(
        self, space_view: NDArray[np.int16], tetromino: NDArray[np.bool], iteration: int
    ) -> bool:
        windows = np.lib.stride_tricks.sliding_window_view(space_view, tetromino.shape)
        for window_index in np.ndindex(windows.shape[:2]):
            if np.any(np.logical_and(windows[window_index], tetromino)) or not self.space_fillable(
                space_view, to_be_placed_tetromino=PositionedTetromino(window_index, tetromino)
            ):
                continue

            y, x = window_index
            height, width = tetromino.shape

            space_view[y : y + height, x : x + width] = np.where(
                tetromino,
                tetromino * (iteration + 1),
                space_view[y : y + height, x : x + width],
            )
            return True

        return False

    # TODO: use check_around argument to make this check more efficient, validating only the area around a newly placed
    # block
    @staticmethod
    def space_fillable(
        space_view: NDArray[np.int16], to_be_placed_tetromino: PositionedTetromino | None = None
    ) -> bool:
        island_map = np.where(space_view == 0, False, True)

        if to_be_placed_tetromino:
            y, x = to_be_placed_tetromino.position
            height, width = to_be_placed_tetromino.tetromino.shape
            island_map[y : y + height, x : x + width] = np.logical_or(
                to_be_placed_tetromino.tetromino,
                island_map[y : y + height, x : x + width],
            )

        return all(island_size % 4 == 0 for island_size in TetrominoSpaceFiller._generate_islands(island_map))

    @staticmethod
    def _generate_islands(island_map: NDArray[np.int16]) -> Iterator[int]:
        for index, cell in np.ndenumerate(island_map):
            if cell == 0:
                yield TetrominoSpaceFiller._flood_fill_island(island_map, index)

    @staticmethod
    def _flood_fill_island(island_map: NDArray[np.int16], index: tuple[int, int]) -> int:
        island_size = 0
        to_visit = [index]

        while to_visit:
            y, x = to_visit.pop()
            if not island_map[y, x]:
                island_size += 1
                island_map[y, x] = True
                to_visit.extend(
                    idx
                    for idx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
                    if TetrominoSpaceFiller._in_bounds(island_map.shape, idx) and not island_map[idx]
                )

        return island_size

    @staticmethod
    def _in_bounds(shape: tuple[int, int], index: tuple[int, int]) -> bool:
        return 0 <= index[0] < shape[0] and 0 <= index[1] < shape[1]
