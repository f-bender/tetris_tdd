import logging
import time
from functools import partial

import click
import numpy as np

from tetris.cli.helpers import generate_space, parse_holes
from tetris.space_filling_coloring import drawer
from tetris.space_filling_coloring.concurrent_fill_and_colorize import fill_and_colorize
from tetris.space_filling_coloring.four_colorizer import FourColorizer, UnableToColorizeError
from tetris.space_filling_coloring.fuzz_test_concurrent_fill_and_colorize import fuzz_test as run_fill_fuzz_test
from tetris.space_filling_coloring.tetromino_space_filler import NotFillableError, TetrominoSpaceFiller

LOGGER = logging.getLogger(__name__)

# TODO refine fuzz_test_concurrent_fill_and_colorize.py into a more generic module that lets you define spaces to be
# filled and/or colorized, and have the fuzz-test entrypoint be THIS here instead of the current fuzz_test file
# which shall be renamed after being made more general)
# Note: maybe (probably) still okay to have a fuzz_test function in that module that is called from here


@click.command()
@click.option("--width", type=int, required=True, help="Width of the space.")
@click.option("--height", type=int, required=True, help="Height of the space.")
@click.option("--holes", type=str, default=None, help='Define holes as "y1,x1,y2,x2;y1,x1,y2,x2;...".')
@click.option("--inverted", is_flag=True, default=False, help="Invert space (fill around holes).")
@click.option("--seed", type=int, default=None, help="Seed for RNG during filling/coloring.")
@click.option("--use-rng/--no-rng", default=True, show_default=True, help="Use randomness.")
@click.option(
    "--top-left-tendency/--no-top-left-tendency", default=True, show_default=True, help="Bias filler towards top-left."
)
@click.option("--colorize/--no-colorize", default=True, show_default=True, help="Color the filled space (4-coloring).")
@click.option(
    "--minimum-separation-steps", type=int, default=0, show_default=True, help="Min steps between filling and coloring."
)
@click.option(
    "--allow-coloring-retry/--no-allow-coloring-retry",
    default=True,
    show_default=True,
    help="Allow colorizer to retry once if it fails.",
)
@click.option("--draw/--no-draw", default=True, show_default=True, help="Draw process iteratively in terminal.")
# Removed --output-file option
@click.option("--fuzz-test", is_flag=True, default=False, help="Run the dedicated space filling fuzz tester.")
def fill_space(
    width: int,
    height: int,
    holes: str | None,
    inverted: bool,
    seed: int | None,
    use_rng: bool,
    top_left_tendency: bool,
    colorize: bool,
    minimum_separation_steps: int,
    allow_coloring_retry: bool,
    draw: bool,
    fuzz_test: bool,
) -> None:
    """Fill (and optionally color) a 2D space with tetrominos."""
    if fuzz_test:
        _run_fuzz_test(draw)
        return  # Exit after fuzz test

    # --- Normal Operation ---
    LOGGER.info("Starting space filling/coloring process.")
    start_time = time.monotonic()
    try:
        hole_list = parse_holes(holes)
        initial_space_bool = generate_space(width, height, hole_list, inverted)
        initial_space_int = initial_space_bool.astype(np.int32) - 1  # -1 holes, 0 fillable

        if not TetrominoSpaceFiller.space_can_be_filled(initial_space_int):
            msg = "Generated space configuration cannot be filled (island sizes not divisible by 4)."
            LOGGER.error(msg)
            raise ValueError(msg)  # Or click.ClickException

        if colorize:
            final_filled_space, final_colored_space = _fill_and_colorize_space(
                initial_space_bool, use_rng, seed, minimum_separation_steps, allow_coloring_retry, draw
            )
        else:
            final_filled_space = _fill_space_only(initial_space_int, use_rng, seed, top_left_tendency, draw)
            final_colored_space = None  # Ensure it's defined

        end_time = time.monotonic()
        LOGGER.info("Filling complete in %.2f seconds.", end_time - start_time)

        _validate_results(final_filled_space, final_colored_space, colorize)

    except (NotFillableError, UnableToColorizeError, ValueError) as e:
        LOGGER.exception("Failed to fill/color space: %s", e)
        # Optionally draw the failed state if possible and meaningful
        # (Might need access to the intermediate state before the error)
    except Exception:
        LOGGER.exception("An unexpected error occurred during space filling/coloring.")
        # traceback.print_exc() # Logger should capture this
    finally:
        LOGGER.info("Fill space command finished.")


def _run_fuzz_test(draw: bool) -> None:
    """Executes the dedicated fuzz testing function."""
    LOGGER.warning("Starting Space Filling FUZZ TEST mode!")
    LOGGER.info("This will run indefinitely with random parameters. Press Ctrl+C to stop.")
    try:
        # Call the fuzz test function directly
        run_fill_fuzz_test(draw=draw)  # Pass draw option
    except KeyboardInterrupt:
        LOGGER.info("Fuzz testing stopped by user.")
    except Exception:
        LOGGER.exception("An error occurred during fuzz testing.")
        # traceback.print_exc() # Logger should capture this


def _fill_and_colorize_space(
    initial_space_bool: np.ndarray, use_rng: bool, seed: int | None, min_sep_steps: int, allow_retry: bool, draw: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Handles the combined filling and coloring process."""
    LOGGER.info("Starting concurrent fill and colorize process.")
    color_generator = fill_and_colorize(
        initial_space_bool,
        use_rng=use_rng,
        rng_seed=seed,
        minimum_separation_steps=min_sep_steps,
        allow_coloring_retry=allow_retry,
    )
    filled_space, colored_space = np.array([[]]), np.array([[]])  # Init for safety
    for filled_space, colored_space in color_generator:
        if draw:
            # Draw combined view: color where available, else filled block index
            draw_array = np.where(colored_space > 0, colored_space, filled_space)
            drawer.draw_array_fancy(draw_array)
            # Add a small sleep if drawing is too fast for the user to see
            # time.sleep(0.01) # Example: 10ms delay
    # Get final state after StopIteration (returned value)
    final_filled_space, final_colored_space = color_generator.value  # type: ignore
    return final_filled_space, final_colored_space


def _fill_space_only(
    initial_space_int: np.ndarray, use_rng: bool, seed: int | None, top_left_tendency: bool, draw: bool
) -> np.ndarray:
    """Handles filling the space without coloring."""
    LOGGER.info("Starting fill-only process.")

    # Define callback only if drawing
    callback = None
    if draw:
        # Use partial to pass the array that will be modified inplace
        # Note: This draws AFTER the update.
        callback = partial(drawer.draw_array_fancy, initial_space_int)

    space_filler = TetrominoSpaceFiller(
        initial_space_int,  # Operates inplace
        use_rng=use_rng,
        rng_seed=seed,
        top_left_tendency=top_left_tendency,
        space_updated_callback=callback,
    )

    # If drawing via iterator is preferred over callback:
    # if draw and callback is None:
    #     fill_iterator = space_filler.ifill()
    #     for _ in fill_iterator:
    #         drawer.draw_array_fancy(space_filler.space)
    #         # time.sleep(0.01) # Optional delay
    # else:
    #     # Drawing via callback or not drawing at all
    #     space_filler.fill()

    # Simpler: rely on callback if draw=True, otherwise just fill.
    space_filler.fill()
    return space_filler.space


def _validate_results(filled_space: np.ndarray | None, colored_space: np.ndarray | None, was_colorized: bool) -> None:
    """Validates the final filled and optionally colored space."""
    if filled_space is None or filled_space.size == 0:
        LOGGER.warning("Final filled space is not available for validation.")
        return

    try:
        LOGGER.info("Validating filled space...")
        TetrominoSpaceFiller.validate_filled_space(filled_space)
        if was_colorized:
            if colored_space is None or colored_space.size == 0:
                LOGGER.warning("Final colored space is not available for validation.")
            else:
                LOGGER.info("Validating colored space...")
                FourColorizer.validate_colored_space(colored_space, filled_space)
        LOGGER.info("Validation successful.")
    except ValueError as e:
        LOGGER.exception("Validation Failed: %s", e)
        # Optionally draw the invalid final state
        # Note: This requires the state *at the time of failure*, which isn't
        # easily available here. A simple print might be better.
        LOGGER.exception("Final (invalid) filled state:\n%s", np.array2string(filled_space, threshold=np.inf))
        if was_colorized and colored_space is not None:
            LOGGER.exception("Final (invalid) colored state:\n%s", np.array2string(colored_space, threshold=np.inf))
