import contextlib
import inspect
import sys
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from tetris.exceptions import BaseTetrisError
from tetris.space_filling_coloring.offset_iterables import CyclingOffsetIterable, RandomOffsetIterable


class UnableToColorizeError(BaseTetrisError):
    pass


class FourColorizer:
    STACK_FRAMES_SAFETY_MARGIN = 10
    NUM_COLORS = 4
    UNCOLORABLE_MESSAGE = "Wasn't able to color the space with 4 colors, this should not be possible!"

    class UncolorableBlockError(BaseTetrisError):
        pass

    def __init__(  # noqa: PLR0913
        self,
        space: NDArray[np.int32],
        *,
        cycle_offset: bool = True,
        use_rng: bool = True,
        rng_seed: int | None = None,
        space_updated_callback: Callable[[], None] | None = None,
        total_blocks_to_color: int | None = None,
        closeness_threshold: int = 6,
    ) -> None:
        """Initialize the FourColorizer.

        Args:
            space: The space to be colored. It will not be modified inplace. Values <= 0 are ignored and will not be
                colored. Values > 0 are considered spots to be colored. All cells with the same value will be colored
                the same. There should be no gaps in the contained values, i.e. if np.max(space) == N, then all values
                in range(1, N + 1) should be contained in the space.
                The resulting space will only have values from 0 to 4. It will be 0 wherever space <= 0, and 1 to 4
                everywhere else, representing the four colors.
            cycle_offset: Whether to cycle through different starting offsets when choosing which color to use for a
                block. If True, the first attempt to color each block is a different color. If False, all blocks are
                first attempted to be colored with color 1, then 2, and so on. Defaults to True.
            use_rng: Whether to use randomness in the color offset selection. Ignored if cycle_offset is False, as this
                color offset selection is the only place where randomness is potentially applied. Defaults to True.
            rng_seed: Optional seed to use for all RNG. Only applies, and should only be specified if cycle_offset
                and use_rng are True. Ignored if use_rng or cycle_offset is False.
            space_updated_callback: Callback function which is called each time after the colored space has been updated
                (a block has been colored or uncolored). This effectively temporarily hands control back to the user of
                this class, letting it act one the latest space update (e.g. for drawing).
            total_blocks_to_color: The total number of blocks that need to be colored. If not provided, it is
                determined from the given `space` array. A higher value than would be automatically determined can be
                specified in order to indicate that the `space` array will change concurrently, and the colorization
                will only be considered finished after this number of blocks has been colored.
            closeness_threshold: The maximum manhattan distance between two blocks for them to be considered "close" to
                each other. This is used as a heuristic to determine whether the change/removal of a block could make
                another (previously uncolorable) block colorable while backtracking. 1.5 * the largest occurring blocks
                seems to be a reasonable value. High values may severely degrade performance due to many unnecessary
                backtracks. Low values may make the algorithm fail. Defaults to 6, assuming that blocks have size 4.
                Change at your own risk!
        """
        max_value = int(np.max(space))

        if total_blocks_to_color is not None and total_blocks_to_color < max_value:
            msg = "total_blocks_to_color must be at least as large as the current maximum value in the space!"
            raise ValueError(msg)

        self._total_blocks_to_color = total_blocks_to_color if total_blocks_to_color is not None else max_value

        if self._total_blocks_to_color <= 0:
            msg = f"{self.total_blocks_to_color =}; there is nothing to be colored!"
            raise ValueError(msg)

        if gaps := (set(range(1, max_value + 1)) - set(np.unique(space[space > 0]))):
            msg = f"Space must contain every value from 1 up to its max value! It has gaps at {gaps}."
            raise ValueError(msg)

        self._space_to_be_colored = space
        self._colored_space = np.zeros_like(space, dtype=np.uint8)

        self._num_colored_blocks = 0
        self._space_updated_callback = space_updated_callback

        self._color_selection_iterable: Iterable[int] = list(range(1, self.NUM_COLORS + 1))
        if cycle_offset:
            if use_rng:
                self._color_selection_iterable = RandomOffsetIterable(self._color_selection_iterable, seed=rng_seed)
            else:
                self._color_selection_iterable = CyclingOffsetIterable(self._color_selection_iterable)

        self._uncolorable_block: int | None = None
        self._finished = False
        self._closeness_threshold = closeness_threshold
        # Keep track of all blocks that already have neighbors with 3 different colors, i.e. there is only one option
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

    @property
    def finished(self) -> bool:
        return self._finished

    @contextlib.contextmanager
    def _ensure_sufficient_recursion_depth(self) -> Iterator[None]:
        required_depth = len(inspect.stack()) + self._total_blocks_to_color + self.STACK_FRAMES_SAFETY_MARGIN

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

        assert self._finished, self.UNCOLORABLE_MESSAGE

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

        if not self._finished:
            self._uncolorable_block = None
            raise UnableToColorizeError(self.UNCOLORABLE_MESSAGE)

        return self._colored_space

    def _icolorize(self) -> Iterator[None]:
        for block_was_placed in self._generate_coloring():
            yield

            if self._any_colored_block_no_longer_exists():
                # continue in order to avoid even yielding, which would give space_to_be_colored the opportunity to
                # change even further which we don't want to allow!
                continue

            if not block_was_placed:
                # _generate_coloring has yielded without coloring a block.
                # This means that there is currently no block left to be colored in space_to_be_colored.
                # We assume that space_to_be_colored is being mutated concurrently outside this generator, so we
                # yield in order to wait for the next mutation which we hope will add a new block that can then be
                # colored.
                continue

            yield from self._icolorize()
            if self._finished:
                # when finished, simply unwind the stack all the way to the top, *without* backtracking
                return

            if self._any_colored_block_no_longer_exists():
                # continue in order to avoid even yielding, which would give space_to_be_colored the opportunity to
                # change even further which we don't want to allow!
                continue

            yield

    @contextlib.contextmanager
    def _single_option_block_restorer(self) -> Iterator[None]:
        single_option_blocks_copy = self._single_option_blocks.copy()
        try:
            yield
        finally:
            self._single_option_blocks = single_option_blocks_copy

    def _generate_coloring(self) -> Iterator[bool]:
        """Yields a bool, whether a block was colorized or not."""
        with self._single_option_block_restorer():
            while (block_to_colorize := self._get_next_block_to_colorize()) is None:
                yield False

                if self._any_colored_block_no_longer_exists():
                    return

            yield from self._generate_coloring_for_block(block_to_colorize)

    def _generate_coloring_for_block(self, block_to_colorize: int) -> Iterator[bool]:  # noqa: C901, PLR0912
        """Yields a bool, whether a block was colorized or not."""
        neighboring_colors, neighboring_uncolored_blocks = self._get_neighboring_colors_and_uncolored_blocks(
            block_to_colorize
        )

        block_positions_index = self._space_to_be_colored == block_to_colorize

        for color in self._color_selection_iterable:
            if color in neighboring_colors:
                continue

            assert np.all(self._colored_space[block_positions_index] == 0), (
                "Trying to color an already colored block! This should never happen!\n"
                f"{np.argwhere(block_positions_index) = }\n"
                f"{self._colored_space[block_positions_index] = }\n"
                f"{self._space_to_be_colored[block_positions_index] = }\n"
            )

            self._colored_space[block_positions_index] = color
            self._num_colored_blocks += 1

            try:
                new_single_option_neighbors = [
                    block
                    for block in neighboring_uncolored_blocks
                    if self._has_only_one_color_option(block) and block not in self._single_option_blocks
                ]
            except self.UncolorableBlockError:
                # if any of the neighboring blocks has not even a single color option, we are in a dead end and need to
                # undo this colorization
                self._num_colored_blocks -= 1
                self._colored_space[block_positions_index] = 0
                continue

            self._single_option_blocks.extend(new_single_option_neighbors)

            if self._space_updated_callback is not None:
                self._space_updated_callback()

            if self._num_colored_blocks == self._total_blocks_to_color:
                self._finished = True
                return

            yield True

            invalid_state_before_uncolorizing_block = self._any_colored_block_no_longer_exists()

            self._num_colored_blocks -= 1
            self._colored_space[block_positions_index] = 0

            if self._space_updated_callback is not None:
                self._space_updated_callback()

            for _ in range(len(new_single_option_neighbors)):
                self._single_option_blocks.pop()

            if self._any_colored_block_no_longer_exists():
                # even after removing this block, we are still in an invalid state, so we need to return immediately
                # NOTE: empirically, it looks like removing this conditional return doesn't break the algorithm
                return

            if self._uncolorable_block is not None:
                if self._uncolorable_block not in self._space_to_be_colored:
                    # uncolorable_block is no longer part of the space to be colored -> unset it
                    self._uncolorable_block = None
                elif not self._blocks_are_close(block_to_colorize, self._uncolorable_block):
                    # If the block we have just uncolored during backtracking is not close to the uncolorable block,
                    # then this change has likely not made the uncolorable block colorable again, thus we need to
                    # immediately fast backtrack further.
                    return

            if invalid_state_before_uncolorizing_block:
                # We were in an invalid state before uncolorizing the block, but are in a valid state now (otherwise we
                # would have returned earlier). I.e. the uncolorizing of this block has put us into a valid state again.
                # Since this backtrack was caused by an external influence, this doesn't constitute a dead-end, and
                # this block should once again consider all possible coloring options ("try again").
                self._generate_coloring_for_block(block_to_colorize)
                return

        # at this point we either have tried everything to color `self._next_block_to_color` but were unsuccessful,
        # or are in the process of fast backtracking because we have encountered an uncolorable block deeper down
        # the stack

        # in case we are not already in the process of fast backtracking to a block we were unable to color
        if not self._uncolorable_block:
            # remember this block as the block that could not be colored, then fast backtrack until a block close to
            # it is uncolored, hopefully removing the issue that made this one uncolorable
            self._uncolorable_block = block_to_colorize

    def _has_only_one_color_option(self, block: int) -> bool:
        """Check if there is only exactly one color that could be given to the block.

        Specifically, check if exactly 3 of the 4 possible colors are already taken by one of the block's neighbors.

        Raises:
            UncolorableBlockError: If the block has no color options, meaning that it's impossible to colors this block,
                i.e. we are in a dead end and need to backtrack.

        Returns:
            Bool, whether the block has only exactly one color option.
        """
        num_neighboring_colors = len(self._get_neighboring_colors_and_uncolored_blocks(block)[0])

        if num_neighboring_colors == self.NUM_COLORS:
            raise self.UncolorableBlockError

        return num_neighboring_colors == self.NUM_COLORS - 1

    def _any_colored_block_no_longer_exists(self) -> bool:
        """Check if any of the blocks we have already colored now no longer exists in _space_to_be_colored.

        If this is the case, we are in an invalid state that should be rectified ASAP by backtracking until we are in a
        valid state again.
        """
        return bool(np.any(np.logical_and(self._space_to_be_colored <= 0, self._colored_space > 0)))

    def _get_next_block_to_colorize(self) -> int | None:
        # Prio 1: Try to color the uncolorable block if there is one
        if self._uncolorable_block is not None:
            block_to_colorize = self._uncolorable_block
            self._uncolorable_block = None
            if block_to_colorize in self._single_option_blocks:
                self._single_option_blocks.remove(block_to_colorize)
            return block_to_colorize

        # Prio 2: Try to color the first single option block
        if self._single_option_blocks:
            return self._single_option_blocks.popleft()

        # Prio 3: Try to color the uncolored block with the smallest ID value
        blocks_left_to_be_colored = self._space_to_be_colored[
            np.logical_and(self._colored_space == 0, self._space_to_be_colored > 0)
        ]
        if blocks_left_to_be_colored.size == 0:
            return None

        return int(np.min(blocks_left_to_be_colored))

    def _get_neighboring_colors_and_uncolored_blocks(self, block: int) -> tuple[set[int], set[int]]:
        neighboring_colors: set[int] = set()
        neighboring_uncolored_blocks: set[int] = set()

        block_positions = np.argwhere(self._space_to_be_colored == block)

        for offset in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            for neighbor_position in block_positions + offset:
                if (
                    not 0 <= neighbor_position[0] < self._colored_space.shape[0]
                    or not 0 <= neighbor_position[1] < self._colored_space.shape[1]
                ):
                    # neighbor is out of bounds
                    continue

                neighboring_block = self._space_to_be_colored[*neighbor_position]
                if neighboring_block == block or neighboring_block <= 0:
                    # neighbor is part of the block itself (not a "real" neighbor)
                    # or neighbor is not part of a block, but of the background that shall remain untouched
                    continue

                if (neighboring_color := self._colored_space[*neighbor_position]) != 0:
                    # neighbor is colored
                    neighboring_colors.add(int(neighboring_color))
                else:
                    # neighbor is uncolored
                    neighboring_uncolored_blocks.add(int(neighboring_block))

        return neighboring_colors, neighboring_uncolored_blocks

    def _blocks_are_close(self, block1: int, block2: int) -> bool:
        # In concurrent operation (concurrent coloring while space is being filled), sometimes block1 == block2.
        # This is unexpected and it's unclear why exactly this happens, but empirically,
        # everything still ends up working fine and the algorithm ends up being successful, with a correct solution.
        # So we don't assert block1 != block2 here.
        # (Note: to gain a better understanding what's going on, it could still make sense to debug this situation)

        block1_positions = np.argwhere(self._space_to_be_colored == block1)
        block2_positions = np.argwhere(self._space_to_be_colored == block2)
        if block1_positions.size == 0 or block2_positions.size == 0:
            return False

        return any(
            np.min(np.sum(np.abs(block2_positions - block1_position), axis=1)) <= self._closeness_threshold
            for block1_position in block1_positions
        )

    @staticmethod
    def validate_colored_space(colored_space: NDArray[np.uint8], space_being_colored: NDArray[np.int32]) -> None:
        """Validate that the given colored space is a valid four-coloring of the given space to be colored.

        Raises:
            ValueError: If the colored space is invalid.
        """
        if np.any(colored_space < 0):
            msg = "Colored space contains negative values."
            raise ValueError(msg)

        if np.any(colored_space > FourColorizer.NUM_COLORS):
            msg = f"Colored space contains values greater than {FourColorizer.NUM_COLORS = }."
            raise ValueError(msg)

        if np.any(colored_space[space_being_colored <= 0] != 0):
            msg = "Colored space contains values for non-existent blocks."
            raise ValueError(msg)

        for block_index in np.unique(space_being_colored):
            if block_index <= 0:
                continue

            block_colors = np.unique(colored_space[space_being_colored == block_index])
            if len(block_colors) != 1:
                msg = f"Block {block_index} has multiple colors ({block_colors.tolist()})"
                raise ValueError(msg)

            block_color = block_colors[0]
            if any(
                colored_space[neighbor_position] == block_color
                for neighbor_position in FourColorizer._neighboring_positions(space_being_colored, block_index)
            ):
                msg = f"Block {block_index} has a neighbor with the same color ({block_color})"
                raise ValueError(msg)

    @staticmethod
    def _neighboring_positions(space_being_colored: NDArray[np.int32], block_index: int) -> set[tuple[int, int]]:
        block_positions = np.argwhere(space_being_colored == block_index)
        return {
            tuple(neighbor_position)
            for offset in ((-1, 0), (0, -1), (1, 0), (0, 1))
            for neighbor_position in block_positions + offset
            if neighbor_position not in block_positions
            and (
                0 <= neighbor_position[0] < space_being_colored.shape[0]
                and 0 <= neighbor_position[1] < space_being_colored.shape[1]
            )
        }
