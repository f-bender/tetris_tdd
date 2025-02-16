import logging
import pickle
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from pprint import pformat
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.heuristic_bot_gym.evaluator import Evaluator
from tetris.heuristic_bot_gym.evaluators.parallel_within_bot import ParallelWithinBotEvaluator
from tetris.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


class DetailedHeuristicEvaluator:
    def __init__(
        self,
        num_games: int = 50,
        seed: int = 42,
        evaluator: Evaluator | None = None,
        report_file: Path = Path(__file__).parent / "report.csv",
    ) -> None:
        self._num_games = num_games
        self._seed = seed
        # smaller than normal board size by default to make evaluation not take so long
        self._evaluator = evaluator or ParallelWithinBotEvaluator(board_size=(15, 10))
        self._report_file = report_file

        self._report_df = (
            pd.read_csv(report_file)
            if report_file.is_file()
            else pd.DataFrame(
                columns=[
                    "mean_score",
                    "median_score",
                    "max_score",
                    "min_score",
                    "evaluation_seed",
                    "num_games",
                    "evaluator_type",
                    "evaluator_config",
                    "heuristic",
                    "scores",
                ]
            )
        )
        if len(self._report_df) > 0:
            LOGGER.info("Current best mean score: %f", self._report_df["mean_score"].max())

    def evaluate(self, heuristic: Heuristic, seed: int | None = None) -> None:
        seed = self._seed if seed is None else seed
        LOGGER.info("Evaluating '%r' with seed %d", heuristic, seed)

        rng = random.Random(seed)
        evaluator_seeds = [rng.randrange(2**32) for _ in range(self._num_games)]

        scores = self._evaluator(heuristics=[heuristic] * self._num_games, seeds=evaluator_seeds)

        self._report_scores(heuristic=heuristic, scores=scores, seed=seed)

    def _report_scores(self, heuristic: Heuristic, scores: list[float], seed: int) -> None:
        heuristic_stats = {
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "evaluation_seed": seed,
            "num_games": self._num_games,
            "evaluator_type": f"{type(self._evaluator).__module__}.{type(self._evaluator).__qualname__}",
            "evaluator_config": repr(self._evaluator.config),
            "heuristic": repr(heuristic),
            "scores": scores,
        }
        LOGGER.info("Evaluation results: %s", pformat(heuristic_stats))

        self._report_df.loc[len(self._report_df)] = heuristic_stats
        self._report_df.to_csv(self._report_file, index=False)

    def evaluate_from_checkpoints(self, checkpoints: Iterable[Path], top_k: int = 10) -> None:
        checkpoints = list(checkpoints)
        for k in range(top_k):
            for checkpoint in checkpoints:
                heuristics = pickle.load(checkpoint.open("rb"))["population"]  # noqa: S301
                if len(heuristics) <= k:
                    continue

                heuristic = heuristics[k]

                if len(self._report_df) > 0 and repr(heuristic) in self._report_df["heuristic"].to_numpy():
                    LOGGER.debug("Skipping '%r' from '%s' since it has already been evaluated", heuristic, checkpoint)
                    continue

                LOGGER.debug("Evalauting heusitic %d from '%s'", k + 1, checkpoint)
                self.evaluate(heuristic)


def main() -> None:
    configure_logging()

    DetailedHeuristicEvaluator().evaluate_from_checkpoints(
        sorted(
            (
                checkpoint
                for checkpoint in (Path(__file__).parent / ".checkpoints").glob("**/*.pkl")
                if checkpoint.is_file()
                and datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=ZoneInfo("Europe/Berlin"))
                > datetime(2025, 2, 9, 0, 45, tzinfo=ZoneInfo("Europe/Berlin"))
            ),
            key=lambda file: file.stat().st_mtime,
            reverse=True,
        )
    )


if __name__ == "__main__":
    main()
