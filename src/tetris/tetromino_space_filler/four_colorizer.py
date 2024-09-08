import contextlib
import inspect
import sys
from collections import deque
from collections.abc import Callable, Generator, Iterator
from itertools import chain

import numpy as np
from numpy.typing import NDArray

from tetris.tetromino_space_filler.offset_iterables import CyclingOffsetIterable, RandomOffsetIterable


class FourColorizer:
    STACK_FRAMES_SAFETY_MARGIN = 10
    NUM_COLORS = 4

    def __init__(
        self,
        space: NDArray[np.int32],
        *,
        use_rng: bool = True,
        rng_seed: int | None = None,
        space_updated_callback: Callable[[], None] | None = None,
        total_blocks_to_color: int | None = None,
    ) -> None:
        """Initialize the four-colorer.

        Args:
            space: The space to be colored. It will not be modified inplace. Values <= 0 are ignored and will not be
                colored. Values > 0 are considered spots to be colored. All cells with the same value will be colored
                the same. There should be no gaps in the contained values, i.e. if np.max(space) == N, then all values
                in range(1, N + 1) should be contained in the space.
                The resulting space will only have values from 0 to 4. It will be 0 wherever space <= 0, and 1 to 4
                everywhere else, representing the four colors.

            TODO: the following has yet to be adapted

            use_rng: Whether to use randomness in the selection of color, selection of tetromino rotation/transpose,
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

        if np.max(space) <= 0:
            msg = "Space must contain at least one positive value. Otherwise there is nothing to be colored!"
            raise ValueError(msg)

        #! for some reason, this number doesn't match up with the actual recursion depth, or with
        #! self._num_colored_blocks after the algorithm has finished. FIND OUT WHY!
        #! Do we ever maybe color over an element that has already been colored?
        max_value = int(np.max(space))

        if gaps := (set(range(1, max_value + 1)) - set(np.unique(space[space > 0]))):
            msg = f"Space must contain every value from 1 up to its max value! It has gaps at {gaps}."
            raise ValueError(msg)

        self._space_to_be_colored = space
        self._colored_space = np.zeros_like(space, dtype=np.uint8)

        if total_blocks_to_color is not None and total_blocks_to_color < max_value:
            msg = "total_blocks_to_color must be at least as large as the current maximum value in the space!"
            raise ValueError(msg)

        self._total_blocks_to_color = total_blocks_to_color or max_value
        self._num_colored_blocks = 0
        self._space_updated_callback = space_updated_callback

        colors = list(range(1, self.NUM_COLORS + 1))
        self._color_selection_iterable = (
            RandomOffsetIterable(items=colors, seed=rng_seed) if use_rng else CyclingOffsetIterable(colors)
        )

        self._uncolorable_block: int | None = None
        self._finished = False
        # keep track of all blocks that already have neighbors with 3 different colors, i.e. there is only one option
        # when choosing a color for them. Color these with first priority, and only turn to other blocks when this list
        # is empty
        # Every time after coloring a new block, check all its uncolored neighbors and add them to this list if they
        # have only one option
        # add using .append, pop using .popleft
        self._single_option_blocks: deque[int] = deque()

    @property
    def colored_space(self) -> NDArray[np.uint8]:
        return self._colored_space

    @property
    def total_blocks_to_color(self) -> int:
        return self._total_blocks_to_color

    @property
    def num_colored_blocks(self) -> int:
        return self._num_colored_blocks

    @contextlib.contextmanager
    def _ensure_sufficient_recursion_depth(self) -> Iterator[None]:
        required_depth = len(inspect.stack()) + self._total_blocks_to_color + self.STACK_FRAMES_SAFETY_MARGIN + 50

        prev_depth = sys.getrecursionlimit()
        sys.setrecursionlimit(max(required_depth, prev_depth))
        try:
            yield
        finally:
            sys.setrecursionlimit(prev_depth)

    def colorize(self) -> NDArray[np.uint8]:
        """Fill up the space with tetrominos, each identified by a unique number, inplace."""
        with self._ensure_sufficient_recursion_depth():
            self._colorize()

        assert self._finished, "Wasn't able to color the space with 4 colors, this should not be possible!"

        return self._colored_space

    def _colorize(self) -> None:
        for _ in self._generate_coloring():
            self._colorize()
            if self._finished:
                # when finished, simply unwind the stack all the way to the top, *without* backtracking
                return

    def icolorize(self) -> Generator[None, None, NDArray[np.uint8]]:
        """Fill up the space with tetrominos, each identified by a unique number, inplace."""
        with self._ensure_sufficient_recursion_depth():
            yield from self._icolorize()

        assert self._finished, "Wasn't able to color the space with 4 colors, this should not be possible!"

        return self._colored_space

    def _icolorize(self) -> Iterator[None]:
        for _ in self._generate_coloring():
            yield

            yield from self._icolorize()
            if self._finished:
                # when finished, simply unwind the stack all the way to the top, *without* backtracking
                return

            yield

        # self._colored_space[self._space_to_be_colored == self._next_block_to_color] = 5
        # yield

    def _generate_coloring(self) -> Iterator[None]:
        is_single_option_block = False
        if self._uncolorable_block is not None:
            block_to_colorize = self._uncolorable_block
            self._uncolorable_block = None
        elif self._single_option_blocks:
            is_single_option_block = True
            block_to_colorize = self._single_option_blocks.popleft()
            # print("single", block_to_colorize, self._single_option_blocks)
        else:
            block_to_colorize = int(
                np.min(
                    self._space_to_be_colored[np.logical_and(self._colored_space == 0, self._space_to_be_colored > 0)]
                )
            )
            # print("multi ", block_to_colorize)

        neighboring_colors, neighboring_uncolored_blocks = self._get_neighboring_colors_and_uncolored_blocks(
            block=block_to_colorize
        )

        for color in self._color_selection_iterable:
            if color in neighboring_colors:
                continue

            # if self._next_block_to_color == self._uncolorable_block:
            #     self._uncolorable_block = None

            # if not np.all(self._colored_space[self._space_to_be_colored == block_to_colorize] == 0):
            #     msg = "This should not happen!"
            #     #! but it does happen... find out why!
            #     #! probably an issue with bookkeeping of _single_option_blocks...
            #     raise RuntimeError(msg)

            self._colored_space[self._space_to_be_colored == block_to_colorize] = color
            self._num_colored_blocks += 1

            # if (
            #     self._uncolorable_block is not None
            #     and len(self._get_neighboring_colors_and_uncolored_blocks(self._uncolorable_block)) == self.NUM_COLORS
            # ):
            #     continue

            # x = False
            # for uncolored_block in neighboring_uncolored_blocks:
            #     if len(self._get_neighboring_colors_and_uncolored_blocks(uncolored_block)[0]) == self.NUM_COLORS:
            #         # self._colored_space[self._space_to_be_colored == uncolored_block] = 6
            #         self._next_block_to_color -= 1
            #         self._colored_space[self._space_to_be_colored == self._next_block_to_color] = 0
            #         x = True
            #         break
            # if x:
            #     continue

            no_option_neighbor = False
            single_option_neighbors: list[int] = []
            for neighboring_uncolored_block in neighboring_uncolored_blocks:
                num_neighboring_colors = len(
                    self._get_neighboring_colors_and_uncolored_blocks(neighboring_uncolored_block)[0]
                )
                if num_neighboring_colors == self.NUM_COLORS:
                    self._num_colored_blocks -= 1
                    self._colored_space[self._space_to_be_colored == block_to_colorize] = 0
                    no_option_neighbor = True
                    break

                if (
                    num_neighboring_colors == self.NUM_COLORS - 1
                    and neighboring_uncolored_block not in self._single_option_blocks
                ):
                    single_option_neighbors.append(neighboring_uncolored_block)

            if no_option_neighbor:
                continue

            self._single_option_blocks.extend(single_option_neighbors)

            # if any(
            #     len(self._get_neighboring_colors_and_uncolored_blocks(uncolored_block)[0]) == self.NUM_COLORS
            #     for uncolored_block in neighboring_uncolored_blocks
            # ):
            #     self._next_block_to_color -= 1
            #     self._colored_space[self._space_to_be_colored == self._next_block_to_color] = 0
            #     continue

            if self._space_updated_callback is not None:
                self._space_updated_callback()

            # print(f"{self._num_colored_blocks}")
            # print(self._colored_space)
            # print(self._single_option_blocks)

            # if self._num_colored_blocks == self._total_blocks_to_color:
            if np.all(self._colored_space[self._space_to_be_colored > 0]):
                self._finished = True
                return

            yield

            self._num_colored_blocks -= 1
            self._colored_space[self._space_to_be_colored == block_to_colorize] = 0

            if self._space_updated_callback is not None:
                self._space_updated_callback()

            for _ in range(len(single_option_neighbors)):
                self._single_option_blocks.pop()

            if self._uncolorable_block is not None and not self._blocks_are_close(
                self._space_to_be_colored, block_to_colorize, self._uncolorable_block
            ):
                # If the block we have just uncolored during backtracking is not adjacent to the uncolorable block, then
                # this change has not made the uncolorable block colorable again, thus we need to immediately fast
                # backtrack further.
                if is_single_option_block:
                    self._single_option_blocks.appendleft(block_to_colorize)
                return
            # print("adjacent")

        if is_single_option_block:
            self._single_option_blocks.appendleft(block_to_colorize)

        # at this point we either have tried everything to color `self._next_block_to_color` but were unsuccessful,
        # or are in the process of fast backtracking because we have encountered an uncolorable block deeper down the
        # stack

        # in case we are not already in the process of fast backtracking to a block we were unable to color
        if not self._uncolorable_block:
            # remember this block as the block that could not be colored, then fast backtrack until a block close to it
            # is uncolored, hopefully removing the issue that made this one uncolorable
            self._uncolorable_block = block_to_colorize

    def _get_neighboring_colors_and_uncolored_blocks(self, block: int) -> tuple[set[int], set[int]]:
        neighboring_colors: set[int] = set()
        neighboring_uncolored_blocks: set[int] = set()

        block_positions = np.argwhere(self._space_to_be_colored == block)

        for neighbor_position in chain.from_iterable(
            block_positions + neighbor_offset for neighbor_offset in ((-1, 0), (0, -1), (1, 0), (0, 1))
        ):
            if (
                not 0 <= neighbor_position[0] < self._colored_space.shape[0]
                or not 0 <= neighbor_position[1] < self._colored_space.shape[1]
                or self._space_to_be_colored[*neighbor_position] == block
            ):
                # neighbor is out of bounds, or is part of the block itself (not a "real" neighbor)
                continue

            if self._colored_space[*neighbor_position] != 0:
                # neighbor is colored
                neighboring_colors.add(int(self._colored_space[*neighbor_position]))
            else:
                # neighbor is uncolored
                neighboring_uncolored_blocks.add(int(self._space_to_be_colored[*neighbor_position]))

        return neighboring_colors, neighboring_uncolored_blocks

    @staticmethod
    def _blocks_are_close(space: NDArray[np.int32], block1: int, block2: int) -> bool:
        assert block1 != block2

        block1_positions = np.argwhere(space == block1)
        block2_positions = np.argwhere(space == block2)

        return any(
            np.min(np.sum(np.abs(block2_positions - block1_position), axis=1)) <= 5
            for block1_position in block1_positions
        )
