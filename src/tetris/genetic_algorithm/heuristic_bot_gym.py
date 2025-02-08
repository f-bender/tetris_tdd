import logging
import pickle
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from tetris import logging_config
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import UI
from tetris.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategyImpl
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.ui.cli.ui import CLI

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging_config.configure_logging()

    HeuristicGym(50, ui=CLI()).run()


# TODO make this flexible enough that my multiprocessing idea works with it as well
# or, you know, identify the multiple responsibilities and split them up
class HeuristicGym:
    def __init__(  # noqa: PLR0913
        self,
        population_size: int = 100,
        *,
        ui: UI | None = None,
        max_evaluation_frames: int = 100_000,
        board_size: tuple[int, int] = (20, 10),
        mutator: Callable[[Heuristic], Heuristic] = mutated_heuristic,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
    ) -> None:
        self._ui = ui
        if self._ui:
            self._ui.initialize(*board_size, num_boards=population_size)

        self._max_evaluation_frames = max_evaluation_frames
        self._mutator = mutator
        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._heuristic_bot_controllers: list[HeuristicBotController] = []
        self._games: list[Game] = []
        # come up with a better way than keeping track of these separately?
        self._score_trackers: list[TrackScoreCallback] = []
        self._spawn_strategies: list[SpawnStrategyImpl] = []

        self._create_games(num_games=population_size, board_size=board_size)

    def _create_games(self, num_games: int, board_size: tuple[int, int]) -> None:
        for idx in range(num_games):
            DEPENDENCY_MANAGER.current_game_index = idx

            board = Board.create_empty(*board_size)
            controller = HeuristicBotController(board, lightning_mode=True)
            track_score_callback = TrackScoreCallback()
            spawn_strategy = SpawnStrategyImpl()

            self._heuristic_bot_controllers.append(controller)
            self._score_trackers.append(track_score_callback)
            self._spawn_strategies.append(spawn_strategy)
            self._games.append(
                Game(
                    board=board,
                    controller=controller,
                    rule_sequence=RuleSequence(
                        (
                            MoveRule(),
                            RotateRule(),
                            SpawnDropMergeRule(
                                spawn_strategy=spawn_strategy,
                                speed_strategy=SpeedStrategyImpl(base_interval=10),
                                spawn_delay=0,
                            ),
                            ClearFullLinesRule(),
                        )
                    ),
                    callback_collection=CallbackCollection((track_score_callback,)),
                )
            )

        DEPENDENCY_MANAGER.wire_up(games=self._games)

    def run(self, initial_population: list[Heuristic] | None = None, num_generations: int | None = None) -> None:
        initial_population = (
            initial_population or ([Heuristic()] + [self._mutator(Heuristic()) for _ in range(len(self._games) - 1)])  # type: ignore[call-arg]
        )

        GeneticAlgorithm(
            initial_population=initial_population,
            population_evaluator=self._evaluate_heuristics,
            mutator=self._mutator,
            post_evaluation_callback=self._post_evaluation_callback,
        ).run(num_generations)

    def _evaluate_heuristics(self, heuristics: list[Heuristic]) -> list[float]:
        for controller, heuristic in zip(self._heuristic_bot_controllers, heuristics, strict=True):
            controller.heuristic = heuristic

        for game in self._games:
            game.reset()

        seed = random.randrange(2**32)
        for spawn_strategy in self._spawn_strategies:
            spawn_strategy.select_block_fn = SpawnStrategyImpl.truly_random_select_fn(seed)

        self._run_games()

        return [tracker.score for tracker in self._score_trackers]

    def _run_games(self) -> None:
        num_alive_games = len(self._games)

        for i in range(self._max_evaluation_frames):
            if i % 1000 == 0:
                LOGGER.debug("Frame %d", i)

            for game in self._games:
                if not game.alive:
                    continue

                try:
                    game.advance_frame()
                except GameOverError:
                    num_alive_games -= 1

            if self._ui:
                self._ui.advance_startup()
                self._ui.draw(game.board.as_array() for game in self._games)

            if num_alive_games == 0:
                break

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

    def _save_checkpoint(self, population: list[Heuristic], file: BinaryIO) -> None:
        pickle.dump(
            {
                "population": population,
                "max_evaluation_frames": self._max_evaluation_frames,
                "mutator": self._mutator,
                "board_size": self._games[0].board.size,
            },
            file,
        )

    @classmethod
    def continue_from_latest_checkpoint(
        cls, checkpoint_dir: Path = Path(__file__).parent / ".checkpoints", ui: UI | None = None
    ) -> None:
        checkpoint_files_sorted = sorted(checkpoint_dir.iterdir(), key=lambda file: file.stat().st_mtime, reverse=True)
        latest_checkpoint_file = next(
            (checkpoint_file for checkpoint_file in checkpoint_files_sorted if checkpoint_file.is_file()), None
        )

        if latest_checkpoint_file is None:
            msg = f"checkpoint_dir '{checkpoint_dir}' contains no checkpoint files!"
            raise ValueError(msg)
        logging.debug(latest_checkpoint_file)

        cls.continue_from_checkpoint(latest_checkpoint_file.open("rb"), ui=ui, checkpoint_dir=checkpoint_dir)

    @classmethod
    def continue_from_checkpoint(
        cls,
        checkpoint_file: BinaryIO,
        ui: UI | None = None,
        checkpoint_dir: Path | None = Path(__file__).parent / ".checkpoints",
    ) -> None:
        # make sure to only load from trusted sources
        checkpoint = pickle.load(checkpoint_file)  # noqa: S301

        gym = cls(
            population_size=len(checkpoint["population"]),
            max_evaluation_frames=checkpoint["max_evaluation_frames"],
            board_size=checkpoint["board_size"],
            mutator=checkpoint["mutator"],
            ui=ui,
            checkpoint_dir=checkpoint_dir,
        )

        gym.run(initial_population=checkpoint["population"])


if __name__ == "__main__":
    main()
