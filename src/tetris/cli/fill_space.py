import logging
import random
import re
from pathlib import Path
from typing import TypedDict, cast

import click

from tetris.cli.common import BoardSize
from tetris.space_filling_coloring.fill_and_colorize_config import (
    ColorizeConcurrently,
    ColorizeSequentially,
    DontColorize,
    FillColorizeConfig,
    fuzz_test_fill_and_colorize,
)

LOGGER = logging.getLogger(__name__)


class Hole(click.ParamType):
    name = "hole"

    _REGEX = r"^(\d+),(\d+)-(\d+),(\d+)$"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        match = re.match(self._REGEX, value)
        if not match:
            msg = "Expected two 2D points in the format 'y1,x1-y2,x2', e.g. '30,50-40,60'."
            raise click.BadParameter(msg)

        y1, x1, y2, x2 = map(int, match.groups())

        return (y1, x1), (y2, x2)


class CommonOptions(TypedDict):
    """Common options for the fill-space command."""

    size: tuple[int, int] | None
    hole: tuple[tuple[tuple[int, int], tuple[int, int]]]
    invert: bool
    seed: int | None
    rng: bool
    top_left_tendency: bool
    draw: bool
    error_report_file: Path


@click.group()
@click.option(
    "--size",
    type=BoardSize(),
    default=None,
    help="Size of the space (height and width separated by 'x'). Defaults to the size of the terminal.",
)
@click.option(
    "--hole",
    "-h",
    type=Hole(),
    multiple=True,
    help="Hole in the space (not to be filled), specified as 'y1,x1-y2,x2'. Can be provided multiple times.",
)
@click.option(
    "--invert", "-i", is_flag=True, default=False, help="Invert space (holes are being filled, space around it isn't)."
)
@click.option(
    "--rng/--no-rng",
    default=True,
    show_default=True,
    help="Whether to use randomness. When disabled, filling is deterministic.",
)
@click.option(
    "--seed", type=int, default=None, help="Seed for RNG during filling/coloring. Ignored if --no-rng is set."
)
@click.option(
    "--top-left-tendency/--no-top-left-tendency",
    default=True,
    show_default=True,
    help='Bias filler towards top-left. More efficient because "surface area" is minimized.',
)
@click.option("--draw/--no-draw", default=True, show_default=True, help="Draw process iteratively in terminal.")
@click.option(
    "--error-report-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(__file__).parents[3] / "logs" / "fill_space" / "fuzz_test_errors.log",
    help="File to write error reports to.",
)
@click.pass_context
def fill_space(ctx: click.Context, /, **common_options: CommonOptions) -> None:
    """Fill (and optionally color) a 2D space with tetrominos."""
    ctx.obj = common_options


@fill_space.command()
@click.option("--max-holes", type=int, default=10, show_default=True, help="Maximum number of holes to generate.")
@click.pass_context
def fuzz_test(ctx: click.Context, max_holes: int) -> None:
    """Continuously generate random spaces, and fill and colorize them.

    Arguments --hole, --invert, --rng, --seed, and --top-left-tendency are ignored. Instead, randomness is always used,
    and the space filling config is generated randomly every time. --size is used as the max size of the space to be
    generated.
    """
    common_options = cast("CommonOptions", ctx.obj)

    fuzz_test_fill_and_colorize(
        max_size=common_options["size"],
        max_holes=max_holes,
        draw=common_options["draw"],
        error_report_file=common_options["error_report_file"],
    )


@fill_space.command()
@click.pass_context
def no_color(ctx: click.Context) -> None:
    """Fill the space with tetrominos without coloring them."""
    common_options = cast("CommonOptions", ctx.obj)

    seed = common_options["seed"] or random.randrange(2**32)

    try:
        config = FillColorizeConfig.create_and_coerce_to_make_fillable(
            size=common_options["size"],
            holes=common_options["hole"],
            fill_and_colorize_rng_seed=seed if common_options["rng"] else None,
            inverted=common_options["invert"],
            top_left_tendency=common_options["top_left_tendency"],
            mode=DontColorize(),
        )
    except ValueError as e:
        raise click.BadArgumentUsage(str(e)) from e

    config.execute_and_check(draw=common_options["draw"], error_report_file=common_options["error_report_file"])


@fill_space.group()
def color() -> None:
    """Fill and color the space with tetrominos."""


@color.command()
@click.pass_context
def subsequent(ctx: click.Context) -> None:
    """First fully fill the space with tetrominos, then fully color them."""
    common_options = cast("CommonOptions", ctx.obj)

    seed = common_options["seed"] or random.randrange(2**32)

    try:
        config = FillColorizeConfig.create_and_coerce_to_make_fillable(
            size=common_options["size"],
            holes=common_options["hole"],
            fill_and_colorize_rng_seed=seed if common_options["rng"] else None,
            inverted=common_options["invert"],
            top_left_tendency=common_options["top_left_tendency"],
            mode=ColorizeSequentially(),
        )
    except ValueError as e:
        raise click.BadArgumentUsage(str(e)) from e

    config.execute_and_check(draw=common_options["draw"], error_report_file=common_options["error_report_file"])


@color.command()
@click.option(
    "--minimum-separation-steps",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Minimum number of steps that the filling algorithm shall stay ahead of the coloring algorithm. "
        "Ignored if no colorization is done (--no-colorize) or it's done non-concurrently (--no-concurrent)."
    ),
)
@click.option(
    "--allow-coloring-retry/--no-allow-coloring-retry",
    default=True,
    show_default=True,
    help=(
        "Allow colorizer to retry if it fails when running concurrently with space filling. "
        "Ignored if no colorization is done (--no-colorize) or it's done non-concurrently (--no-concurrent)."
    ),
)
@click.pass_context
def concurrent(ctx: click.Context, *, minimum_separation_steps: int, allow_coloring_retry: bool) -> None:
    """Interleave space filling and coloring steps, filling and coloring the space concurrently."""
    common_options = cast("CommonOptions", ctx.obj)

    seed = common_options["seed"] or random.randrange(2**32)

    try:
        config = FillColorizeConfig.create_and_coerce_to_make_fillable(
            size=common_options["size"],
            holes=common_options["hole"],
            fill_and_colorize_rng_seed=seed if common_options["rng"] else None,
            inverted=common_options["invert"],
            top_left_tendency=common_options["top_left_tendency"],
            mode=ColorizeConcurrently(
                minimum_separation_steps=minimum_separation_steps,
                allow_coloring_retry=allow_coloring_retry,
            ),
        )
    except ValueError as e:
        raise click.BadArgumentUsage(str(e)) from e

    config.execute_and_check(draw=common_options["draw"], error_report_file=common_options["error_report_file"])
