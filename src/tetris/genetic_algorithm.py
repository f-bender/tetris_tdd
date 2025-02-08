import logging
from collections.abc import Callable
from itertools import count, cycle

import numpy as np

LOGGER = logging.getLogger(__name__)


class GeneticAlgorithm[T]:
    def __init__(  # noqa: PLR0913
        self,
        initial_population: list[T],
        population_evaluator: Callable[[list[T]], list[float]],
        mutator: Callable[[T], T],
        *,
        post_evaluation_callback: Callable[[list[T], list[float]], None] | None = None,
        survival_rate: float = 0.5,
        elitism_factor: float = 1.0,
    ) -> None:
        self._population = initial_population
        self._population_evaluator = population_evaluator
        self._mutator = mutator
        self._post_evaluation_callback = post_evaluation_callback

        self._survival_rate = survival_rate
        self._elitism_factor = elitism_factor

    def run(self, num_generations: int | None = None) -> None:
        for i in range(num_generations) if num_generations is not None else count():
            LOGGER.info("Evaluating generation %d", i)

            fitnesses = self._population_evaluator(self._population)

            if self._post_evaluation_callback is not None:
                self._post_evaluation_callback(self._population, fitnesses)

            self._evolve(fitnesses)

    def _evolve(self, fitnesses: list[float]) -> None:
        """Evolve the population by selecting survivors and mutating them.

        Expects self._population to be sorted by fitness.
        """
        population_size = len(self._population)
        assert len(fitnesses) == population_size

        survivors = self._select_survivors(fitnesses)
        self._population = survivors

        for i in cycle(range(len(survivors))):
            if len(self._population) >= population_size:
                break

            self._population.append(self._mutator(survivors[i]))

    def _select_survivors(self, fitnesses: list[float]) -> list[T]:
        """Select survivors from a sorted population using weighted sampling.

        Args:
            fitnesses: Fitness values of the individuals in the population.
            self._survival_rate: Fraction of the population that should survive. Must be in the range [0, 1].
            self._elitism_factor: How much to favor the best individuals.
                A factor of 0 means that all individuals have the same chance of surviving.
                A very high factor (approaching infinity) means that exactly the best
                `round(survival_rate * len(self._population))` individuals survive.
        """
        population_size = len(self._population)
        num_survivors = min(max(1, round(population_size * self._survival_rate)), population_size)

        # individuals with higher fitness have a higher weight (chance of surviving)
        weights = np.array(fitnesses, dtype=float)
        # ensure all weights are > 0
        weights -= min(np.min(weights) - 1e-10, 0)
        weights **= self._elitism_factor

        probabilities = weights / weights.sum()

        selected_indices = np.random.default_rng().choice(
            population_size, size=num_survivors, replace=False, p=probabilities
        )

        return [self._population[i] for i in selected_indices]
