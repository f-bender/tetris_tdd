import logging
from pathlib import Path
from typing import cast

import click
import pandas as pd

from tetris.cli.common import HeuristicParam, evaluator_options
from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.heuristic_gym.detailed_heuristic_evaluator import DetailedHeuristicEvaluator
from tetris.heuristic_gym.evaluators.evaluator import EvaluatorImpl

LOGGER = logging.getLogger(__name__)


@click.group()
@click.option("--seed", type=int, default=42, show_default=True, help="Seed for evaluation randomness.")
@click.option(
    "--num-games", type=click.IntRange(min=1), default=50, show_default=True, help="Number of games for evaluation."
)
@evaluator_options()
@click.option(
    "--report-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default="./data/evaluation_reports/report.csv",
    show_default=True,
    help="CSV file to save results to.",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    seed: int,
    num_games: int,
    evaluator: EvaluatorImpl,
    report_file: Path,
) -> None:
    """Evaluate a Heuristic for an automatic tetris bot, creating a detailed performance report."""
    # detailed evaluator is passed on to the subcommands via ctx
    ctx.obj = DetailedHeuristicEvaluator(num_games=num_games, seed=seed, evaluator=evaluator, report_file=report_file)


@evaluate.command()
@click.argument(
    "checkpoints",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    nargs=-1,
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="How many of the top Heuristics to evaluate per checkpoint file.",
)
@click.pass_context
def from_train_checkpoint(ctx: click.Context, checkpoints: tuple[Path, ...], top_k: int) -> None:
    """Evaluate top-performing Heuristics from one or more training checkpoint files."""
    if not checkpoints:
        msg = "At least one checkpoint file must be specified."
        raise click.BadParameter(msg)

    cast("DetailedHeuristicEvaluator", ctx.obj).evaluate_from_checkpoints(checkpoints=checkpoints, top_k=top_k)


@evaluate.command()
@click.argument("report-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--top-k",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help=(
        "How many of the top Heuristics (according to each performance column) to evaluate from the report file. "
        "Set to 0 to re-evaluate *all* Heuristics in the report file."
    ),
)
@click.option(
    "--performance-column",
    "-c",
    type=click.Choice(["mean_score", "median_score", "max_score", "min_score"]),
    default=["mean_score", "median_score", "max_score", "min_score"],
    multiple=True,
    show_default=True,
    help="Columns to choose the top performers of. Can specify multiple. Ignored if top-k is set to 0.",
)
@click.pass_context
def from_evaluation_report(
    ctx: click.Context, report_file: Path, top_k: int, performance_column: tuple[str, ...]
) -> None:
    """Re-evaluate the top-performing Heuristics of a previous evaluation report file."""
    detailed_evaluator = cast("DetailedHeuristicEvaluator", ctx.obj)
    report_df = pd.read_csv(report_file)

    if top_k == 0:
        detailed_evaluator.reevaluate_from_report(report_df)
    else:
        detailed_evaluator.reevaluate_top_performers(
            report_df=report_df, top_k=top_k, performance_columns=performance_column
        )


@evaluate.command()
@click.argument("heuristic", type=HeuristicParam(), default="Heuristic()")
@click.pass_context
def explicit(ctx: click.Context, heuristic: Heuristic) -> None:
    """Evaluate an explicitly specified Heuristic, specified via its string representation.

    The Heuristic string representation is easily copy-pasteable from report files or log outputs. Example:
    "Heuristic(sum_of_cell_heights_close_to_top_weight=1, num_distinct_overhangs_weight=14, num_rows_with_overhung_holes_weight=5, num_overhung_cells_weight=0.4, num_overhanging_cells_weight=0.04, num_narrow_gaps_weight=20, sum_of_cell_heights_weight=0.04, sum_of_adjacent_height_differences_weight=2, close_to_top_threshold=2)".
    It defaults to the so far best performing heuristic.
    """  # noqa: E501
    cast("DetailedHeuristicEvaluator", ctx.obj).evaluate(heuristic)
