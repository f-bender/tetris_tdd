import functools
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import click

from tetris.cli.common import HeuristicParam, evaluator_options, genetic_algorithm_options
from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.game_logic.rules.core.spawn.spawn import SpawnRule
from tetris.genetic_algorithm import GeneticAlgorithm
from tetris.heuristic_gym.evaluators.evaluator import EvaluatorImpl
from tetris.heuristic_gym.heuristic_gym import HeuristicGym

LOGGER = logging.getLogger(__name__)


class _CommonOptions(TypedDict):
    population_size: int
    checkpoint_dir: Path
    generations: int | None


@click.group()
@click.option(
    "--population-size",
    "-n",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Number of heuristics per generation. Default: 50 when training from scratch, otherwise the population size as "
        "read from the checkpoint file."
    ),
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default="./data/train_checkpoints",
    show_default=True,
    help="Directory for storing checkpoints after each generation.",
)
@click.option(
    "--generations", type=click.IntRange(min=1), default=None, help="Number of generations to run (default: infinite)."
)
@click.pass_context
def train(ctx: click.Context, /, **common_options: _CommonOptions) -> None:
    """Train Heuristics for an automatic tetris bot using a genetic algorithm."""
    ctx.obj = common_options


@train.command()
@evaluator_options()
@genetic_algorithm_options()
@click.option(
    "--seed-heuristic",
    type=HeuristicParam(),
    default="Heuristic()",
    show_default=True,
    help=(
        "String representation for the first Heuristic to create the initial population from by creating mutations of "
        "it. "
        "The Heuristic string representation is easily copy-pasteable from report files or log outputs. Example:\n"
        "Heuristic(sum_of_cell_heights_close_to_top_weight=1, num_distinct_overhangs_weight=14, num_rows_with_overhung_holes_weight=5, num_overhung_cells_weight=0.4, num_overhanging_cells_weight=0.04, num_narrow_gaps_weight=20, sum_of_cell_heights_weight=0.04, sum_of_adjacent_height_differences_weight=2, close_to_top_threshold=2)\n"  # noqa: E501
        "Defaults to the so far best performing heuristic."
    ),
)
@click.pass_context
def from_scratch(
    ctx: click.Context,
    evaluator: EvaluatorImpl,
    genetic_algorithm: GeneticAlgorithm[Heuristic],
    seed_heuristic: Heuristic,
) -> None:
    common_options = cast("_CommonOptions", ctx.obj)
    HeuristicGym(
        population_size=common_options["population_size"] or 50,
        evaluator=evaluator,
        genetic_algorithm=genetic_algorithm,
        checkpoint_dir=common_options["checkpoint_dir"],
    ).run(initial_population=[seed_heuristic], num_generations=common_options["generations"])


@train.command()
@click.argument(
    "checkpoint-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, required=False
)
@evaluator_options(default_none=True, assemble_object=False)
@genetic_algorithm_options(default_none=True, assemble_object=False)
@click.pass_context
def from_checkpoint(  # noqa: PLR0913
    ctx: click.Context,
    checkpoint_file: Path | None,
    # evaluator options
    board_size: tuple[int, int] | None,
    max_frames: int | None,
    block_selection: str | None,
    # genetic algorithm options
    mutation_rate: float | None,
    survival_rate: float | None,
    elitism_factor: float | None,
) -> None:
    common_options = cast("_CommonOptions", ctx.obj)

    kwargs_to_overwrite: dict[str, Any] = {}

    if board_size:
        kwargs_to_overwrite.setdefault("evaluator_config", {})["board_size"] = board_size
    if max_frames:
        kwargs_to_overwrite.setdefault("evaluator_config", {})["max_evaluation_frames"] = max_frames
    if block_selection:
        kwargs_to_overwrite.setdefault("evaluator_config", {})["block_selection_fn_from_seed"] = getattr(
            SpawnRule, f"{block_selection}_selection_fn"
        )

    if mutation_rate:
        kwargs_to_overwrite.setdefault("genetic_algorithm_config", {})["mutation_rate"] = (
            functools.partial(mutated_heuristic, mutation_rate=mutation_rate),
        )
    if survival_rate:
        kwargs_to_overwrite.setdefault("genetic_algorithm_config", {})["survival_rate"] = survival_rate
    if elitism_factor:
        kwargs_to_overwrite.setdefault("genetic_algorithm_config", {})["elitism_factor"] = elitism_factor

    if checkpoint_file is not None:
        HeuristicGym.continue_from_checkpoint(
            checkpoint_file=checkpoint_file.open("rb"),
            checkpoint_dir=common_options["checkpoint_dir"],
            new_population_size=common_options["population_size"],
            num_generations=common_options["generations"],
            **kwargs_to_overwrite,
        )
    else:
        if not common_options["checkpoint_dir"].is_dir():
            msg = "Checkpoint directory must exist if checkpoint_file is not specified!"
            raise click.BadParameter(msg)

        HeuristicGym.continue_from_latest_checkpoint(
            checkpoint_dir=common_options["checkpoint_dir"],
            new_population_size=common_options["population_size"],
            num_generations=common_options["generations"],
            **kwargs_to_overwrite,
        )
