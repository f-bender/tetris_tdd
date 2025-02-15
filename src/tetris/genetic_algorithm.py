import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count, cycle, islice

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneticAlgorithm[T]:
    mutator: Callable[[T], T]
    survival_rate: float = 0.5
    elitism_factor: float = 1.0

    def run(
        self,
        population: list[T],
        population_evaluator: Callable[[list[T]], list[float]],
        fitnesses: list[float] | None = None,
        *,
        num_generations: int | None = None,
        post_evaluation_callback: Callable[[list[T], list[float]], None] | None = None,
    ) -> None:
        if fitnesses:
            if len(fitnesses) != len(population):
                msg = (
                    "`fitnesses` must have same length as `population`, "
                    f"got {len(fitnesses) = } vs. {len(population) = }"
                )
                raise ValueError(msg)

            LOGGER.debug("Fitnesses provided, evolving before first evaluation...")
            # if we have been provided fitnesses of the initial population (e.g. saved from a previous run),
            # immediately use them to evolve the initial population
            sorted_population, sorted_fitnesses = self._to_sorted(population=population, fitnesses=fitnesses)
            population = self._evolve(sorted_population=sorted_population, sorted_fitnesses=sorted_fitnesses)

        for i in range(num_generations) if num_generations is not None else count():
            LOGGER.info("Evaluating generation %d", i)

            fitnesses = population_evaluator(population)

            sorted_population, sorted_fitnesses = self._to_sorted(population=population, fitnesses=fitnesses)

            if post_evaluation_callback is not None:
                post_evaluation_callback(sorted_population, sorted_fitnesses)

            population = self._evolve(sorted_population=sorted_population, sorted_fitnesses=sorted_fitnesses)

    @staticmethod
    def _to_sorted(population: list[T], fitnesses: list[float]) -> tuple[list[T], list[float]]:
        sorted_population, sorted_fitnesses = zip(
            *sorted(
                zip(population, fitnesses, strict=True),
                key=lambda x: x[1],
                reverse=True,
            ),
            strict=True,
        )
        return list(sorted_population), list(sorted_fitnesses)

    def _evolve(self, sorted_population: list[T], sorted_fitnesses: list[float]) -> list[T]:
        """Evolve the population by selecting survivors and mutating them, returning the evolved population."""
        population_size = len(sorted_population)
        assert len(sorted_fitnesses) == population_size

        survivors = self._select_survivors(sorted_population, sorted_fitnesses)
        return self.expanded_population_through_mutation(population=survivors, new_size=population_size)

    def expanded_population_through_mutation(self, population: list[T], new_size: int) -> list[T]:
        """Return a population expanded to the desired size.

        Mutated versions of the existing individuals in the populations are used for filling the list up to the desired
        size.
        """
        assert new_size >= len(population)
        new_population = population.copy()
        new_population.extend(
            self.mutator(individual) for individual in islice(cycle(population), new_size - len(population))
        )
        return new_population

    def _select_survivors(self, sorted_population: list[T], sorted_fitnesses: list[float]) -> list[T]:
        """Select survivors from a sorted population using weighted sampling.

        Args:
            sorted_population: The population of individuals, sorted by fitness.
            sorted_fitnesses: Fitness values of the individuals in the population, sorted.
            self._survival_rate: Fraction of the population that should survive. Must be in the range [0, 1].
            self._elitism_factor: How much to favor the best individuals.
                A factor of 0 means that all individuals have the same chance of surviving.
                A very factor of `inf` means that exactly the best `round(survival_rate * len(self._population))`
                individuals survive.
                A factor of < 0 is not recommended and will lead to less fit individuals having a higher chance of
                surviving.
        """
        population_size = len(sorted_population)
        num_survivors = min(max(1, round(population_size * self.survival_rate)), population_size)

        if self.elitism_factor == float("inf"):
            return sorted_population[:num_survivors]

        # individuals with higher fitness have a higher weight (chance of surviving)
        weights = np.array(sorted_fitnesses, dtype=float)
        # ensure all weights are > 0
        weights -= min(np.min(weights) - 1e-10, 0)
        weights **= self.elitism_factor

        probabilities = weights / weights.sum()

        selected_indices = np.random.default_rng().choice(
            population_size, size=num_survivors, replace=False, p=probabilities
        )

        return [sorted_population[i] for i in sorted(selected_indices)]
