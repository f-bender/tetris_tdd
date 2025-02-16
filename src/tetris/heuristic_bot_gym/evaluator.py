import logging
import random
from abc import ABC, abstractmethod
from typing import Any

from tetris.controllers.heuristic_bot.heuristic import Heuristic

LOGGER = logging.getLogger(__name__)


class Evaluator(ABC):
    def __call__(self, heuristics: list[Heuristic], seeds: list[int] | int | None = None) -> list[float]:
        """Evaluate the heuristics, returning their fitnesses."""
        if seeds is None:
            seeds = random.randrange(2**32)
        if isinstance(seeds, int):
            LOGGER.debug("Evaluating with seed %d", seeds)
            seeds = [seeds] * len(heuristics)

        if len(seeds) != len(heuristics):
            msg = f"`seeds` must have same length as `heuristics`, got {len(seeds) = } vs. {len(heuristics) = }"
            raise ValueError(msg)

        return self._evaluate(heuristics=heuristics, seeds=seeds)

    @abstractmethod
    def _evaluate(self, heuristics: list[Heuristic], seeds: list[int]) -> list[float]:
        """Evaluate the heuristics, returning their fitnesses."""

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
