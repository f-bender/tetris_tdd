import logging
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from pprint import pformat
from typing import Any, BinaryIO

from tetris import logging_config
from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.genetic_algorithm import GeneticAlgorithm
from tetris.utils import deep_merge

LOGGER = logging.getLogger(__name__)


# this should obviously moved out of here eventually
def main() -> None:
    logging_config.configure_logging()

    HeuristicGym(50).run()


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, heuristics: list[Heuristic]) -> list[float]:
        """Evaluate the heuristics, returning their fitnesses."""

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""


class HeuristicGym:
    def __init__(
        self,
        population_size: int = 100,
        evaluator: Evaluator | None = None,
        *,
        mutator: Callable[[Heuristic], Heuristic] = mutated_heuristic,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
    ) -> None:
        self._population_size = population_size

        self._evaluator = evaluator or self._default_evaluator()

        self._mutator = mutator
        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_evaluator() -> Evaluator:
        from tetris.heuristic_bot_gym.evaluators.synchronous import SyncEvaluator

        return SyncEvaluator()

    def run(self, initial_population: Sequence[Heuristic] | None = None, num_generations: int | None = None) -> None:
        initial_population = list(initial_population) if initial_population is not None else [Heuristic()]  # type: ignore[call-arg]

        if len(initial_population) > self._population_size:
            initial_population = initial_population[: self._population_size]
        elif len(initial_population) < self._population_size:
            initial_population = self._expanded_population(initial_population, new_size=self._population_size)

        GeneticAlgorithm(
            initial_population=initial_population,
            population_evaluator=self._evaluator,
            mutator=self._mutator,
            post_evaluation_callback=self._post_evaluation_callback,
        ).run(num_generations)

    def _expanded_population(self, population: list[Heuristic], new_size: int) -> list[Heuristic]:
        assert new_size > len(population)
        new_population = population.copy()
        new_population.extend(
            self._mutator(heuristic) for heuristic in islice(cycle(population), new_size - len(population))
        )
        return new_population

    def _post_evaluation_callback(self, population: list[Heuristic], fitnesses: list[float]) -> None:
        sorted_population, sorted_fitnesses = zip(
            *sorted(
                zip(population, fitnesses, strict=True),
                key=lambda x: x[1],
                reverse=True,
            ),
            strict=True,
        )

        LOGGER.info("Sorted fitnesses: %s", sorted_fitnesses)
        LOGGER.info("Best heuristic '%r' scored %d", sorted_population[0], sorted_fitnesses[0])

        LOGGER.debug("Top heuristics: %r", sorted_population[:10])

        if self._checkpoint_dir:
            self._save_checkpoint(
                sorted_population,
                (
                    self._checkpoint_dir
                    / datetime.now(UTC).strftime(f"%Y-%m-%d_%H-%M-%S_fitness-{sorted_fitnesses[0]}.pkl")
                ).open("wb"),
            )

    def _save_checkpoint(self, population: Sequence[Heuristic], file: BinaryIO) -> None:
        pickle.dump(
            {
                "population": population,
                "mutator": self._mutator,
                "evaluator_type": type(self._evaluator),
                "evaluator_config": self._evaluator.config,
            },
            file,
        )

    @classmethod
    def continue_from_latest_checkpoint(
        cls,
        checkpoint_dir: Path = Path(__file__).parent / ".checkpoints",
        new_population_size: int | None = None,
        **kwargs_to_overwrite: Any,  # noqa: ANN401
    ) -> None:
        checkpoint_files_sorted = sorted(checkpoint_dir.iterdir(), key=lambda file: file.stat().st_mtime, reverse=True)
        latest_checkpoint_file = next(
            (checkpoint_file for checkpoint_file in checkpoint_files_sorted if checkpoint_file.is_file()), None
        )

        if latest_checkpoint_file is None:
            msg = f"checkpoint_dir '{checkpoint_dir}' contains no checkpoint files!"
            raise ValueError(msg)
        logging.debug(latest_checkpoint_file)

        cls.continue_from_checkpoint(
            latest_checkpoint_file.open("rb"),
            checkpoint_dir=checkpoint_dir,
            new_population_size=new_population_size,
            **kwargs_to_overwrite,
        )

    @classmethod
    def continue_from_checkpoint(
        cls,
        checkpoint_file: BinaryIO,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
        new_population_size: int | None = None,
        **kwargs_to_overwrite: Any,  # noqa: ANN401
    ) -> None:
        # make sure to only load from trusted sources
        checkpoint: dict[str, Any] = deep_merge(pickle.load(checkpoint_file), kwargs_to_overwrite)  # noqa: S301

        population: list[Heuristic] = checkpoint.pop("population")
        checkpoint["population_size"] = new_population_size or len(population)

        LOGGER.info("Continuing HeuristicBotGym with config\n%s", pformat(checkpoint))

        checkpoint["evaluator"] = checkpoint.pop("evaluator_type")(**checkpoint.pop("evaluator_config"))

        cls(checkpoint_dir=checkpoint_dir, **checkpoint).run(initial_population=population)


if __name__ == "__main__":
    main()
