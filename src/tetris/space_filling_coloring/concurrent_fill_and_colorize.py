import random
from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray

from tetris.space_filling_coloring.four_colorizer import FourColorizer, UnableToColorizeError
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller


def fill_and_colorize(
    space: NDArray[np.bool], *, use_rng: bool = True, rng_seed: int | None = None, allow_coloring_retry: bool = True
) -> Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]:
    """Concurrently fill a space with tetrominos, and colorize the placed tetrominos with four colors.

    Args:
        space: Boolean numpy array. True values indicate the space to be filled. False values indicate "holes" in the
            space to be left empty.
        use_rng: Whether to use a random number generator for the filling and coloring steps.
        rng_seed: Seed for the random number generator. If None, a random seed will be generated.
        allow_coloring_retry: Whether to allow the colorization step to retry if it fails with an
            UnableToColorizeError. This can happen if the concurrent modification of the space being colored (i.e. it
            being filled with tetrominos) "messes" with the colorization algorithm.

    Yields and Returns:
        Always a reference to the same uint8 numpy array which is being filled with 4 colors (int values 1-4) over time.
    """
    space_being_filled = space.astype(np.int32) - 1

    main_rng = random.Random(rng_seed) if rng_seed is not None else None

    space_filler = TetrominoSpaceFiller(
        space_being_filled,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32) if main_rng is not None else None,
    )
    four_colorizer = FourColorizer(
        space_being_filled,
        total_blocks_to_color=space_filler.total_blocks_to_place,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32) if main_rng is not None else None,
        closeness_threshold=7,  # empirically, there are rare cases where 6 is not enough in concurrent operation
    )

    yield space_filler.space, four_colorizer.colored_space

    space_filling_iterator = space_filler.ifill()
    four_colorizing_iterator = four_colorizer.icolorize()

    # we need to ensure the recursion depths is sufficient even here on the "outside", otherwise space filler finishing
    # would reset it back to a too low value while four_colorizer still needs it (setting the depth on the outside
    # makes it so that the `prev_depth` that is being reset back to is still this sufficient depth)
    with space_filler._ensure_sufficient_recursion_depth():  # noqa: SLF001
        # interleave space filling and colorization steps
        for _ in space_filling_iterator:
            try:
                next(four_colorizing_iterator)
            except StopIteration as e:
                msg = "FourColorizer finished before TetrominoSpaceFiller - this should never happen!"
                raise RuntimeError(msg) from e
            except UnableToColorizeError:
                if allow_coloring_retry:
                    four_colorizing_iterator = four_colorizer.icolorize()
                else:
                    raise

            yield space_filler.space, four_colorizer.colored_space

        # filing is done: finish up colorization
        try:
            for _ in four_colorizing_iterator:
                yield space_filler.space, four_colorizer.colored_space
        except UnableToColorizeError:
            if allow_coloring_retry:
                # retry just once more, otherwise more tries will not help
                for _ in four_colorizer.icolorize():
                    yield space_filler.space, four_colorizer.colored_space
            else:
                raise

        return space_filler.space, four_colorizer.colored_space
