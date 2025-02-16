import logging
from collections.abc import Callable
from typing import Any

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.game_logic.components import Board
from tetris.game_logic.components.block import Block
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import UI
from tetris.heuristic_bot_gym.evaluator import Evaluator
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategyImpl
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.ui.cli.ui import CLI

LOGGER = logging.getLogger(__name__)


class SyncEvaluator(Evaluator):
    def __init__(
        self,
        board_size: tuple[int, int] = (20, 10),
        max_evaluation_frames: int = 100_000,
        block_selection_fn_from_seed: Callable[[int], Callable[[], Block]] = SpawnStrategyImpl.truly_random_select_fn,
        ui_class: type[UI] | None = CLI,
    ) -> None:
        self._ui = None if ui_class is None else ui_class()
        self._board_size = board_size

        self._max_evaluation_frames = max_evaluation_frames
        self._block_selection_fn_from_seed = block_selection_fn_from_seed

        self._initialized = False

    @property
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
        return {
            "board_size": self._board_size,
            "max_evaluation_frames": self._max_evaluation_frames,
            "block_selection_fn_from_seed": self._block_selection_fn_from_seed,
            "ui_class": None if self._ui is None else type(self._ui),
        }

    def _initialize(self, num_games: int) -> None:
        self._initialized = True

        if self._ui:
            self._ui.initialize(*self._board_size, num_boards=num_games)

        self._heuristic_bot_controllers: list[HeuristicBotController] = []
        self._games: list[Game] = []
        # come up with a better way than keeping track of these separately?
        self._score_trackers: list[TrackScoreCallback] = []
        self._spawn_strategies: list[SpawnStrategyImpl] = []

        for idx in range(num_games):
            DEPENDENCY_MANAGER.current_game_index = idx

            board = Board.create_empty(*self._board_size)
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

    def _evaluate(self, heuristics: list[Heuristic], seeds: list[int]) -> list[float]:
        if not self._initialized:
            self._initialize(len(heuristics))

        for controller, heuristic in zip(self._heuristic_bot_controllers, heuristics, strict=True):
            controller.heuristic = heuristic

        for game in self._games:
            game.reset()

        for spawn_strategy, seed in zip(self._spawn_strategies, seeds, strict=True):
            spawn_strategy.select_block_fn = self._block_selection_fn_from_seed(seed)

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
                if isinstance(self._ui, CLI):
                    digits = max(
                        len(f"{len(self._games):,}"),
                        len(f"{self._max_evaluation_frames:,}"),
                    )
                    print(f"Game Over: {len(self._games) - num_alive_games:>{digits}} / {len(self._games):<{digits}}")  # noqa: T201
                    print(f"Frames:    {i + 1:>{digits},} / {self._max_evaluation_frames:<{digits},}")  # noqa: T201

                self._ui.advance_startup()
                self._ui.draw(game.board.as_array() for game in self._games)

            if num_alive_games == 0:
                break
