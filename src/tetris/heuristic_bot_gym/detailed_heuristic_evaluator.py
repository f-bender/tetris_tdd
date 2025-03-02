import ast
import logging
import pickle
import random
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import ClassVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.heuristic_bot_gym.evaluator import Evaluator
from tetris.heuristic_bot_gym.evaluators.parallel_within_bot import ParallelWithinBotEvaluator
from tetris.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

type Primitive = str | bytes | int | float | complex | bool | None
type SimpleContainer[T] = dict[LiteralEvallable, T] | list[T] | set[T] | tuple[T, ...]
type LiteralEvallable = Primitive | SimpleContainer[Primitive | LiteralEvallable]


class DetailedHeuristicEvaluator:
    _REPORT_COLUMNS: ClassVar = [
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
        "extra_info",
    ]

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
            pd.read_csv(report_file) if report_file.is_file() else pd.DataFrame(columns=self._REPORT_COLUMNS)
        )
        if len(self._report_df) > 0:
            LOGGER.info("Current best mean score: %f", self._report_df["mean_score"].max())

    def evaluate(self, heuristic: Heuristic, seed: int | None = None, extra_info: LiteralEvallable = None) -> None:
        seed = self._seed if seed is None else seed

        if self._already_evaluated(heuristic=heuristic, seed=seed):
            LOGGER.debug("Not evaluating '%r' since it has already been evaluated with the same config", heuristic)
            return

        LOGGER.info("Evaluating '%r' with seed %d", heuristic, seed)

        rng = random.Random(seed)
        evaluator_seeds = [rng.randrange(2**32) for _ in range(self._num_games)]

        scores = self._evaluator(heuristics=[heuristic] * self._num_games, seeds=evaluator_seeds)

        self._report_scores(heuristic=heuristic, scores=scores, seed=seed, extra_info=extra_info)

    def _already_evaluated(self, heuristic: Heuristic, seed: int) -> bool:
        config = {
            "evaluation_seed": seed,
            "num_games": self._num_games,
            "evaluator_type": f"{type(self._evaluator).__module__}.{type(self._evaluator).__qualname__}",
            "evaluator_config": self._evaluator_config_repr(),
            "heuristic": repr(heuristic),
        }
        return self._report_df[list(config)].eq(pd.Series(config)).all(axis=1).any()

    def _report_scores(
        self, heuristic: Heuristic, scores: list[float], seed: int, extra_info: LiteralEvallable = None
    ) -> None:
        heuristic_stats = {
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "evaluation_seed": seed,
            "num_games": self._num_games,
            "evaluator_type": f"{type(self._evaluator).__module__}.{type(self._evaluator).__qualname__}",
            "evaluator_config": self._evaluator_config_repr(),
            "heuristic": repr(heuristic),
            "scores": repr(scores),
            "extra_info": extra_info and repr(extra_info),
        }
        LOGGER.info("Evaluation results: %s", pformat(heuristic_stats))

        self._report_df.loc[len(self._report_df)] = heuristic_stats
        self._report_df.to_csv(self._report_file, index=False)

    def _evaluator_config_repr(self) -> str:
        # remove address of selection function from repr to ease identification of same config
        return re.sub(r" at 0x[0-9A-F]+", "", repr(self._evaluator.config))

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

                LOGGER.info("Evaluating heuristic %d from '%s'", k + 1, checkpoint)
                self.evaluate(heuristic, extra_info={"checkpoint": str(checkpoint), "top_k": k + 1})

    def reevaluate_from_report(self, report_df: pd.DataFrame) -> None:
        if set(report_df.columns) != set(self._REPORT_COLUMNS):
            msg = (
                "Provided dataframe is not a valid report "
                f"(expected columns {self._REPORT_COLUMNS}, got {report_df.columns.to_list()})."
            )
            raise ValueError(msg)

        for _, row in report_df.iterrows():
            self.evaluate(
                self._literal_eval_heuristic(row["heuristic"]),
                extra_info={
                    **(ast.literal_eval(row["extra_info"]) if pd.notna(row["extra_info"]) else {}),
                    "previous_mean_score": row["mean_score"],
                    "previous_median_score": row["median_score"],
                    "previous_max_score": row["max_score"],
                    "previous_min_score": row["min_score"],
                },
            )

    def reevaluate_top_performers(
        self,
        report_df: pd.DataFrame,
        top_k: int = 5,
        performance_columns: tuple[str, ...] = ("mean_score", "median_score", "max_score", "min_score"),
    ) -> None:
        reevaluate_df = pd.concat([report_df.nlargest(top_k, col) for col in performance_columns])
        reevaluate_df = reevaluate_df.drop_duplicates()

        LOGGER.info("Evaluating top performers:\n%s", reevaluate_df[[*performance_columns, "heuristic", "extra_info"]])
        self.reevaluate_from_report(reevaluate_df)

    @staticmethod
    def _literal_eval_heuristic(heuristic_repr: str) -> Heuristic:
        """Functionally equivalent to `eval(heuristic_repr)` but without the security risk."""
        heuristic_dict_literal = (
            heuristic_repr.replace("Heuristic(", "{'").replace(")", "}").replace("=", "':").replace(", ", ", '")
        )
        return Heuristic(**ast.literal_eval(heuristic_dict_literal))


# this should obviously moved out of here eventually
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
                and "10x10" not in str(checkpoint)
            ),
            key=lambda file: file.stat().st_mtime,
            reverse=True,
        )
    )

    DetailedHeuristicEvaluator(
        num_games=200,
        seed=13,
        evaluator=ParallelWithinBotEvaluator(board_size=(15, 10), max_evaluation_frames=1_000_000),
        report_file=Path(__file__).parent / "report_best_more_detailed.csv",
    ).reevaluate_top_performers(pd.read_csv(Path(__file__).parent / "report.csv"), top_k=10)

    DetailedHeuristicEvaluator(
        num_games=200,
        seed=69,
        evaluator=ParallelWithinBotEvaluator(board_size=(20, 10), max_evaluation_frames=1_000_000),
        report_file=Path(__file__).parent / "report_best_more_detailed_on_20x10.csv",
    ).reevaluate_top_performers(pd.read_csv(Path(__file__).parent / "report_best_more_detailed.csv"), top_k=5)

    DetailedHeuristicEvaluator(
        num_games=200,
        seed=69,
        evaluator=ParallelWithinBotEvaluator(board_size=(20, 10), max_evaluation_frames=1_000_000),
        report_file=Path(__file__).parent / "report_best_more_detailed_on_20x10.csv",
    ).reevaluate_from_report(pd.read_csv(Path(__file__).parent / "report_best_on_20x10.csv"))


if __name__ == "__main__":
    main()
