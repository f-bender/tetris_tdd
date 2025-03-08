import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypedDict

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.game_logic.components import Board
from tetris.game_logic.components.block import Block
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.heuristic_bot_gym.evaluator import Evaluator
from tetris.heuristic_bot_gym.evaluators.runners.parallel import ParallelRunner
from tetris.heuristic_bot_gym.evaluators.runners.synchronous import SynchronousRunner
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategyImpl
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback

LOGGER = logging.getLogger(__name__)


class Runner(ABC):
    @abstractmethod
    def run_games(
        self,
        games: Sequence[Game],
        max_frames: int,
        interval_callback: Callable[[], None] | None = None,
        game_over_callback: Callable[[int], None] | None = None,
    ) -> None: ...

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""

    @abstractmethod
    def initialize(self, num_games: int, board_size: tuple[int, int]) -> None: ...


class RunnerConfig(TypedDict):
    cls: type[ParallelRunner | SynchronousRunner]
    params: dict[str, Any]


class EvaluatorImpl(Evaluator):
    def __init__(
        self,
        *,
        board_size: tuple[int, int] = (20, 10),
        max_evaluation_frames: int = 100_000,
        block_selection_fn_from_seed: Callable[[int], Callable[[], Block]] = SpawnStrategyImpl.truly_random_select_fn,
        runner: SynchronousRunner | ParallelRunner | RunnerConfig | None = None,
    ) -> None:
        self._board_size = board_size
        self._max_evaluation_frames = max_evaluation_frames
        self._block_selection_fn_from_seed = block_selection_fn_from_seed

        self._runner: SynchronousRunner | ParallelRunner
        match runner:
            case {"cls": runner_class, "params": runner_config}:
                self._runner = runner_class(**runner_config)  # type: ignore[operator]
            case None:
                self._runner = ParallelRunner()
            case _:
                self._runner = runner  # type: ignore[assignment]

        self._initialized = False

    @property
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
        return {
            "board_size": self._board_size,
            "max_evaluation_frames": self._max_evaluation_frames,
            "block_selection_fn_from_seed": self._block_selection_fn_from_seed,
            "runner_config": {
                "cls": type(self._runner),
                "params": self._runner.config,
            },
        }

    def _initialize(self, num_games: int) -> None:
        self._initialized = True

        self._runner.initialize(num_games, self._board_size)

        self._heuristic_bot_controllers: list[HeuristicBotController] = []
        self._games: list[Game] = []
        # come up with a better way than keeping track of these separately?
        self._score_trackers: list[TrackScoreCallback] = []
        self._spawn_strategies: list[SpawnStrategyImpl] = []

        process_pool = (
            ProcessPoolExecutor(self._runner.num_workers) if isinstance(self._runner, ParallelRunner) else None
        )
        for idx in range(num_games):
            DEPENDENCY_MANAGER.current_game_index = idx

            board = Board.create_empty(*self._board_size)
            controller = HeuristicBotController(
                board,
                lightning_mode=True,
                process_pool=process_pool,
                # small performance penalty, but the benefit is reproducibility
                ensure_consistent_behaviour=True,
            )
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

    def _evaluate(self, heuristics: list[Heuristic], seeds: list[int]) -> list[float]:
        if not self._initialized:
            self._initialize(len(heuristics))

        for controller, heuristic in zip(self._heuristic_bot_controllers, heuristics, strict=True):
            controller.heuristic = heuristic

        for game in self._games:
            game.reset()

        for spawn_strategy, seed in zip(self._spawn_strategies, seeds, strict=True):
            spawn_strategy.select_block_fn = self._block_selection_fn_from_seed(seed)

        self._runner.run_games(
            games=self._games,
            max_frames=self._max_evaluation_frames,
            interval_callback=lambda: LOGGER.debug("Scores: %s", [tracker.score for tracker in self._score_trackers]),
            game_over_callback=lambda idx: LOGGER.debug("Game %d scored %d", idx, self._score_trackers[idx].score),
        )

        return [tracker.score for tracker in self._score_trackers]
