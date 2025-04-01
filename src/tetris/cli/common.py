import functools
from collections.abc import Callable
from typing import Any

import click

from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.genetic_algorithm import GeneticAlgorithm
from tetris.heuristic_gym.evaluators.evaluator import EvaluatorImpl


class BoardSize(click.ParamType):
    name = "board_size"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> tuple[int, int]:
        # # TODO not sure if this is required. check it and uncomment if necessary. (for when default=None)
        # if value is None:
        #     return None

        try:
            height, width = map(int, value.split("x"))
        except (TypeError, ValueError) as e:
            msg = "Expected two integers separated by 'x' (e.g. '20x10')."
            raise click.BadParameter(msg) from e

        if height <= 0 or width <= 0:
            msg = "Board height and width need to be positive."
            raise click.BadParameter(msg)

        return height, width


def evaluator_options(*, default_none: bool = False, assemble_object: bool = True) -> Callable[..., Callable[..., Any]]:
    def decorator(command: Callable[..., Any]) -> Callable[..., Any]:
        @click.option(
            "--board-size",
            type=BoardSize(),
            default=None if default_none else "20x10",
            show_default=not default_none,
            help="Height and width of the tetris board, separated by 'x'.",
        )
        @click.option(
            "--max-frames",
            type=click.IntRange(min=1),
            default=None if default_none else 1_000_000,
            show_default=not default_none,
            help=(
                "Maximum number of frames to run the evaluation (per game). "
                'Avoids evaluation running forever in case of a "perfect" Heuristic that never fails.'
            ),
        )
        @click.option(
            "--block-selection",
            type=click.Choice(["truly_random", "from_shuffled_bag"]),
            default=None if default_none else "truly_random",
            show_default=not default_none,
            help="How to choose blocks to spawn in the tetris games the Heuristic is being evaluated on.",
        )
        @functools.wraps(command)
        def wrapper(
            *args: Any,  # noqa: ANN401
            board_size: tuple[int, int] | None,
            max_frames: int | None,
            block_selection: str | None,
            **kwargs: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            if assemble_object:
                if board_size is None or max_frames is None or block_selection is None:
                    msg = (
                        "Since default_none=True and assemble_object=True, "
                        "board-size, max-frames, and block-selection must all be specified."
                    )
                    raise click.BadParameter(msg)
                return command(
                    *args,
                    evaluator=EvaluatorImpl(
                        board_size=board_size,
                        max_evaluation_frames=max_frames,
                        block_selection_fn_from_seed=getattr(SpawnStrategyImpl, f"{block_selection}_selection_fn"),
                    ),
                    **kwargs,
                )

            return command(
                *args,
                board_size=board_size,
                max_frames=max_frames,
                block_selection=block_selection,
                **kwargs,
            )

        return wrapper

    return decorator


def genetic_algorithm_options(*, default_none: bool = False, assemble_object: bool = True) -> Callable[..., Any]:
    def decorator(command: Callable[..., Any]) -> Callable[..., Any]:
        @click.option(
            "--mutation-rate",
            type=click.FloatRange(min=0, min_open=True),
            default=None if default_none else 1.0,
            show_default=not default_none,
        )
        @click.option(
            "--survival-rate",
            type=click.FloatRange(0, 1, min_open=True, max_open=True),
            default=0.5,
            show_default=not default_none,
            help=(
                "Fraction of the population that should survive after each generation. The rest of the population will be"
                "filled up with mutated versions of these survivors."
            ),
        )
        @click.option(
            "--elitism-factor",
            type=click.FloatRange(min=0, min_open=True),
            default=1.0,
            show_default=not default_none,
            help=(
                "How much to favor the best individuals. "
                "A factor of 0 means that all individuals have the same chance of surviving. "
                "A factor of `inf` means that exactly the best `round(survival_rate * len(population_size))` individuals "
                "survive."
            ),
        )
        @functools.wraps(command)
        def wrapper(
            *args: Any,  # noqa: ANN401
            mutation_rate: float | None,
            survival_rate: float | None,
            elitism_factor: float | None,
            **kwargs: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            if assemble_object:
                if mutation_rate is None or survival_rate is None or elitism_factor is None:
                    msg = (
                        "Since default_none=True and assemble_object=True, "
                        "board-size, max-frames, and block-selection must all be specified."
                    )
                    raise click.BadParameter(msg)

                return command(
                    *args,
                    genetic_algorithm=GeneticAlgorithm(
                        mutator=functools.partial(mutated_heuristic, mutation_rate=mutation_rate),
                        survival_rate=survival_rate,
                        elitism_factor=elitism_factor,
                    ),
                    **kwargs,
                )
            return command(
                *args, mutation_rate=mutation_rate, survival_rate=survival_rate, elitism_factor=elitism_factor, **kwargs
            )

        return wrapper

    return decorator


class HeuristicParam(click.ParamType):
    name = "heuristic"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> Heuristic:  # noqa: ARG002
        try:
            return Heuristic.from_repr(value)
        except Exception as e:
            msg = f"Invalid Heuristic representation: {value} {e}"
            raise click.BadParameter(msg) from e
