import random
from collections.abc import Generator
from time import sleep

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from tetris.ansi_extensions import color as colorx
from tetris.ansi_extensions import cursor as cursorx
from tetris.space_filling_coloring.four_colorizer import FourColorizer
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller


def fill_and_colorize(
    space: NDArray[np.bool], use_rng: bool = True, rng_seed: int | None = None
) -> Generator[tuple[NDArray[np.int32], NDArray[np.uint8]], None, tuple[NDArray[np.int32], NDArray[np.uint8]]]:
    """Concurrently fill a space with tetrominos, and colorize the placed tetrominos with four colors.

    Args:
        space: Boolean numpy array. True values indicate the space to be filled. False values indicate "holes" in the
            space to be left empty.

    Yields and Returns:
        Always a reference to the same uint8 numpy array which is being filled with 4 colors (int values 1-4) over time.
    """
    space_being_filled = space.astype(np.int32) - 1

    space_filler = TetrominoSpaceFiller(space_being_filled, use_rng=use_rng, rng_seed=rng_seed)
    four_colorizer = FourColorizer(
        space_being_filled, total_blocks_to_color=space_filler.total_blocks_to_place, use_rng=use_rng, rng_seed=rng_seed
    )

    yield space_filler.space, four_colorizer.colored_space

    space_filling_iterator = space_filler.ifill()
    four_colorizing_iterator = four_colorizer.icolorize()

    # we need to ensure the recursion depths is sufficient even here on the "outside", otherwise space filler finishing
    # would reset it back to a too low value while four_colorizer still needs it (setting the depth on the outside
    # makes it so that the `prev_depth` that is being reset back to is still this sufficient depth)
    with space_filler._ensure_sufficient_recursion_depth():  # noqa: SLF001
        # TODO remove priming, and instead properly handle the situations that currently still cause errors
        # (e.g. space filler backtracking back further than the point where the colorizer currently is)
        # for _ in range(1000):
        #     next(space_filling_iterator)

        # interleave space filling and colorization steps
        for _ in space_filling_iterator:
            try:
                next(four_colorizing_iterator)
            except StopIteration as e:
                msg = "FourColorizer finished before TetrominoSpaceFiller - this should never happen!"
                raise RuntimeError(msg) from e

            yield space_filler.space, four_colorizer.colored_space

        # filing is done: finish up colorization
        for _ in four_colorizing_iterator:
            yield space_filler.space, four_colorizer.colored_space

        return space_filler.space, four_colorizer.colored_space


last_drawn: NDArray[np.int32] | None = None
i = 0


def draw_tetromino_space(space: NDArray[np.int32], force: bool = False) -> None:
    global last_drawn, i
    i += 1
    if i % 1 != 0 and not force:
        return

    rd = random.Random()
    if last_drawn is None:
        print(cursor.goto(1, 1), end="")
        for row in space:
            print(
                "".join(
                    rd.seed(int(val))  # type: ignore[func-returns-value]
                    or colorx.bg.rgb_truecolor(rd.randrange(50, 150), rd.randrange(50, 150), rd.randrange(50, 150))
                    + "  "
                    + color.fx.reset
                    if val > 0
                    else "  "
                    for val in row
                )
            )
    else:
        for y, x in np.argwhere(space != last_drawn):
            print(cursor.goto(y + 1, x * 2 + 1), end="")
            print(
                rd.seed(int(val))
                or colorx.bg.rgb_truecolor(rd.randrange(50, 150), rd.randrange(50, 150), rd.randrange(50, 150))
                + "  "
                + color.fx.reset
                if (val := space[y, x]) > 0
                else "  ",
                end="",
                flush=True,
            )
    print(cursor.goto(space.shape[0] + 1) + cursorx.erase_to_end(""), end="")

    last_drawn = space.copy()


def main() -> None:
    # prev_depth = sys.getrecursionlimit()
    # sys.setrecursionlimit(10000)

    for filled_space, colored_space in fill_and_colorize(np.ones((60, 100), dtype=bool), use_rng=True, rng_seed=5):
        sleep(0.01)
        draw_tetromino_space(np.where(colored_space > 0, colored_space, np.where(filled_space > 0, filled_space, 0)))
    draw_tetromino_space(colored_space)

    # sys.setrecursionlimit(prev_depth)


if __name__ == "__main__":
    main()
