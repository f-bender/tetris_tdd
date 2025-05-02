import logging
import random
from collections.abc import Callable, Generator

import numpy as np
from numpy.typing import NDArray

from tetris.space_filling_coloring.four_colorizer import FourColorizer, UnableToColorizeError
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller

LOGGER = logging.getLogger(__name__)


def fill_and_colorize_concurrently(  # noqa: PLR0913
    space: NDArray[np.bool],
    *,
    use_rng: bool = True,
    rng_seed: int | None = None,
    top_left_tendency: bool = True,
    minimum_separation_steps: int = 0,
    allow_coloring_retry: bool = True,
) -> Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]:
    """Concurrently fill a space with tetrominos, and colorize the placed tetrominos with four colors.

    Args:
        space: Boolean numpy array. True values indicate the space to be filled. False values indicate "holes" in the
            space to be left empty.
        use_rng: Whether to use a random number generator for the filling and coloring steps.
        rng_seed: Seed for the random number generator. If None, a random seed will be generated.
        top_left_tendency: Whether to bias the filling to move towards the top left corner of the space.
        minimum_separation_steps: The minimum number of steps by which the space filling and coloring algorithms have to
            be separated. At least this many blocks will placed but not yet colored at any point in time (except during
            backtracks of the space filling algorithm). By default (0), no such separation is enforced.
        allow_coloring_retry: Whether to allow the colorization step to retry if it fails with an
            UnableToColorizeError. This can happen if the concurrent modification of the space being colored (i.e. it
            being filled with tetrominos) "messes" with the colorization algorithm.

    Yields and Returns:
        Always a reference to the same int32 numpy array (space being filled) and uint8 numpy array (space being
        colorized with 4 colors (int values 1-4) over time).
    """
    space_being_filled = space.astype(np.int32) - 1

    main_rng_seed = rng_seed if rng_seed is not None else random.randrange(2**32)
    main_rng = random.Random(main_rng_seed)

    LOGGER.debug(
        "fill_and_colorize_concurrently:\n"  # noqa: G003
        + (f"{main_rng_seed = }\n" if use_rng else "no rng")
        + f"{minimum_separation_steps = }\n"
        + f"{allow_coloring_retry = }\n"
        + f"space:\n{np.array2string(space_being_filled, threshold=np.inf, max_line_width=np.inf)}\n\n"
    )

    space_filler = TetrominoSpaceFiller(
        space_being_filled,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32),
        top_left_tendency=top_left_tendency,
    )
    four_colorizer = FourColorizer(
        space_being_filled,
        total_blocks_to_color=space_filler.total_blocks_to_place,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32),
    )

    yield space_filler.space, four_colorizer.colored_space

    space_filling_iterator = space_filler.ifill()
    four_colorizing_iterator = four_colorizer.icolorize()

    # we need to ensure the recursion depths is sufficient even here on the "outside", otherwise space filler finishing
    # would reset it back to a too low value while four_colorizer still needs it (setting the depth on the outside
    # makes it so that the `prev_depth` that is being reset back to is still this sufficient depth)
    with space_filler._ensure_sufficient_recursion_depth():  # noqa: SLF001
        previous_num_placed_blocks = space_filler.num_placed_blocks

        # interleave space filling and colorization steps
        for _ in space_filling_iterator:
            separation_steps = space_filler.num_placed_blocks - four_colorizer.num_colored_blocks

            # if a block was removed or changed, we must give the colorizer a chance to react and potentially remove the
            # corresponding colorized block on its side
            block_was_removed_or_changed = space_filler.num_placed_blocks <= previous_num_placed_blocks
            previous_num_placed_blocks = space_filler.num_placed_blocks

            if separation_steps > minimum_separation_steps or block_was_removed_or_changed:
                try:
                    next(four_colorizing_iterator)
                except StopIteration:
                    # FourColorizer finished slightly before TetrominoSpaceFiller:
                    # this can happen in rare instances and can safely be ignored
                    pass
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


def fill_and_colorize_subsequently(
    space: NDArray[np.bool],
    *,
    use_rng: bool = True,
    rng_seed: int | None = None,
    top_left_tendency: bool = True,
    space_updated_callback: Callable[[NDArray[np.int32], NDArray[np.uint8] | None], None] | None = None,
) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
    """Subsequently fill a space with tetrominos, and then colorize the placed tetrominos with four colors.

    Args:
        space: Boolean numpy array. True values indicate the space to be filled. False values indicate "holes" in the
            space to be left empty.
        use_rng: Whether to use a random number generator for the filling and coloring steps.
        rng_seed: Seed for the random number generator. If None, a random seed will be generated.
        top_left_tendency: Whether to bias the filling to move towards the top left corner of the space.
        space_updated_callback: Callback function that is called whenever the space is updated. The callback function
            should take two arguments: the space being filled (int32 numpy array) and the colored space (uint8 numpy
            array) or None.

    Returns:
        Tuple of an int32 numpy array (space being filled) and a uint8 numpy array (space after being colorized with 4
        colors (int values 1-4)).
    """
    space_to_fill = space.astype(np.int32) - 1

    main_rng_seed = rng_seed if rng_seed is not None else random.randrange(2**32)
    main_rng = random.Random(main_rng_seed)

    LOGGER.debug(
        "fill_and_colorize_subsequently:\n"  # noqa: G003
        + (f"{main_rng_seed = }\n" if use_rng else "no rng")
        + f"space:\n{np.array2string(space_to_fill, threshold=np.inf, max_line_width=np.inf)}\n\n"
    )

    space_filler = TetrominoSpaceFiller(
        space_to_fill,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32),
        top_left_tendency=top_left_tendency,
        space_updated_callback=(
            lambda: space_updated_callback(space_to_fill, None) if space_updated_callback else None
        ),
    )
    space_filler.fill()

    four_colorizer = FourColorizer(
        space_to_fill,
        use_rng=use_rng,
        rng_seed=main_rng.randrange(2**32),
        space_updated_callback=(
            lambda: space_updated_callback(space_to_fill, four_colorizer.colored_space)
            if space_updated_callback
            else None
        ),
    )
    four_colorizer.colorize()

    return space_filler.space, four_colorizer.colored_space


def fill(
    space: NDArray[np.bool],
    *,
    use_rng: bool = True,
    rng_seed: int | None = None,
    top_left_tendency: bool = True,
    space_updated_callback: Callable[[NDArray[np.int32]], None] | None = None,
) -> NDArray[np.int32]:
    """Fill a space with tetrominos.

    Args:
        space: Boolean numpy array. True values indicate the space to be filled. False values indicate "holes" in the
            space to be left empty.
        use_rng: Whether to use a random number generator for the filling and coloring steps.
        rng_seed: Seed for the random number generator. If None, a random seed will be generated.
        top_left_tendency: Whether to bias the filling to move towards the top left corner of the space.
        space_updated_callback: Callback function that is called whenever the space is updated. The callback function
            should take one argument: the space being filled (int32 numpy array).

    Returns:
        The filled space as an int32 numpy array.
    """
    space_to_fill = space.astype(np.int32) - 1

    LOGGER.debug(
        "fill:\n"  # noqa: G003
        + (f"{rng_seed = }\n" if use_rng else "no rng")
        + f"space:\n{np.array2string(space_to_fill, threshold=np.inf, max_line_width=np.inf)}\n\n"
    )

    space_filler = TetrominoSpaceFiller(
        space_to_fill,
        use_rng=use_rng,
        rng_seed=rng_seed,
        top_left_tendency=top_left_tendency,
        space_updated_callback=(lambda: space_updated_callback(space_to_fill) if space_updated_callback else None),
    )
    space_filler.fill()

    return space_filler.space
