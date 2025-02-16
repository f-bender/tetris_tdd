import logging
import pickle
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from pprint import pformat
from types import LambdaType
from typing import Any, BinaryIO

from tetris import logging_config
from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.genetic_algorithm import GeneticAlgorithm
from tetris.heuristic_bot_gym.evaluator import Evaluator
from tetris.heuristic_bot_gym.evaluators.parallel_within_bot import ParallelWithinBotEvaluator
from tetris.utils import deep_merge

LOGGER = logging.getLogger(__name__)


# this should obviously moved out of here eventually
def main() -> None:
    logging_config.configure_logging()

    HeuristicGym.continue_from_latest_checkpoint(checkpoint_dir=Path(__file__).parent / ".checkpoints" / "15x10")


class HeuristicGym:
    def __init__(
        self,
        population_size: int = 100,
        *,
        evaluator: Evaluator | None = None,
        genetic_algorithm: GeneticAlgorithm[Heuristic] | None = None,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
    ) -> None:
        if genetic_algorithm and checkpoint_dir and isinstance(genetic_algorithm.mutator, LambdaType):
            msg = (
                "Can't save checkpoints when using a lambda function as mutator - use functools.partial or def instead!"
            )
            raise ValueError(msg)

        self._population_size = population_size
        self._evaluator = evaluator or ParallelWithinBotEvaluator()
        self._genetic_algorithm = genetic_algorithm or GeneticAlgorithm(mutator=mutated_heuristic)

        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        initial_population: list[Heuristic] | None = None,
        initial_population_fitnesses: list[float] | None = None,
        num_generations: int | None = None,
    ) -> None:
        initial_population = initial_population or [Heuristic()]  # type: ignore[call-arg]

        if len(initial_population) > self._population_size:
            initial_population = initial_population[: self._population_size]
            if initial_population_fitnesses:
                initial_population_fitnesses = initial_population_fitnesses[: self._population_size]

        elif len(initial_population) < self._population_size:
            initial_population = self._genetic_algorithm.expanded_population_through_mutation(
                population=initial_population, new_size=self._population_size
            )
            initial_population_fitnesses = None

        self._genetic_algorithm.run(
            population=initial_population,
            population_evaluator=self._evaluator,
            fitnesses=initial_population_fitnesses,
            num_generations=num_generations,
            post_evaluation_callback=self._post_evaluation_callback,
        )

    def _post_evaluation_callback(self, sorted_population: list[Heuristic], sorted_fitnesses: list[float]) -> None:
        LOGGER.info("Sorted fitnesses: %s", sorted_fitnesses)
        LOGGER.info("Best heuristic '%r' scored %d", sorted_population[0], sorted_fitnesses[0])

        LOGGER.debug("Top heuristics: %r", sorted_population[:10])

        if self._checkpoint_dir:
            self._save_checkpoint(
                population=sorted_population,
                fitnesses=sorted_fitnesses,
                file=(
                    self._checkpoint_dir
                    / datetime.now(UTC).strftime(f"%Y-%m-%d_%H-%M-%S_fitness-{sorted_fitnesses[0]}.pkl")
                ).open("wb"),
            )

    def _save_checkpoint(self, population: Sequence[Heuristic], fitnesses: list[float], file: BinaryIO) -> None:
        pickle.dump(
            {
                "population": population,
                "fitnesses": fitnesses,
                "evaluator_type": type(self._evaluator),
                "evaluator_config": self._evaluator.config,
                "genetic_algorithm_config": asdict(self._genetic_algorithm),
            },
            file,
        )

    @classmethod
    def continue_from_latest_checkpoint(
        cls,
        checkpoint_dir: Path = Path(__file__).parent / ".checkpoints",
        new_population_size: int | None = None,
        num_generations: int | None = None,
        **kwargs_to_overwrite: Any,  # noqa: ANN401
    ) -> None:
        checkpoint_files_sorted = sorted(checkpoint_dir.iterdir(), key=lambda file: file.stat().st_mtime, reverse=True)
        latest_checkpoint_file = next(
            (
                checkpoint_file
                for checkpoint_file in checkpoint_files_sorted
                if checkpoint_file.is_file() and checkpoint_file.suffix == ".pkl"
            ),
            None,
        )

        if latest_checkpoint_file is None:
            msg = f"checkpoint_dir '{checkpoint_dir}' contains no checkpoint files!"
            raise ValueError(msg)

        LOGGER.debug("Loading checkpoint from '%s'", latest_checkpoint_file)

        cls.continue_from_checkpoint(
            latest_checkpoint_file.open("rb"),
            checkpoint_dir=checkpoint_dir,
            new_population_size=new_population_size,
            num_generations=num_generations,
            **kwargs_to_overwrite,
        )

    @classmethod
    def continue_from_checkpoint(
        cls,
        checkpoint_file: BinaryIO,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
        new_population_size: int | None = None,
        num_generations: int | None = None,
        **kwargs_to_overwrite: Any,  # noqa: ANN401
    ) -> None:
        # make sure to only load from trusted sources
        checkpoint: dict[str, Any] = deep_merge(pickle.load(checkpoint_file), kwargs_to_overwrite)  # noqa: S301

        population: list[Heuristic] = checkpoint.pop("population")
        fitnesses: list[float] | None = checkpoint.pop("fitnesses", None)

        if fitnesses:
            LOGGER.debug("Loaded fitnesses %s", fitnesses)

        population_size = new_population_size or len(population)

        LOGGER.info("Continuing HeuristicBotGym with config\n%s", pformat(checkpoint))

        checkpoint["evaluator"] = checkpoint.pop("evaluator_type")(**checkpoint.pop("evaluator_config"))
        if genetic_algorithm_config := checkpoint.pop("genetic_algorithm_config", None):
            checkpoint["genetic_algorithm"] = GeneticAlgorithm(**genetic_algorithm_config)
        elif mutator := checkpoint.pop("mutator", None):
            # support for legacy checkpoint format
            checkpoint["genetic_algorithm"] = GeneticAlgorithm(mutator=mutator)

        cls(population_size=population_size, checkpoint_dir=checkpoint_dir, **checkpoint).run(
            initial_population=population,
            initial_population_fitnesses=fitnesses,
            num_generations=num_generations,
        )


if __name__ == "__main__":
    main()
