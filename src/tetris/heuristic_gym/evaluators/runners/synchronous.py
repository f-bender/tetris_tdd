from collections.abc import Callable, Sequence
from typing import Any

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.ui.cli.ui import CLI, DynamicLayerConfig


class SynchronousRunner:
    """A synchronous runner for evaluating Tetris games.

    This runner executes games sequentially in a single thread. It's designed to work with HeuristicBotController in
    lightning mode, and non-process-pool mode.
    Each iteration of the main loop advances all games by one frame and then draws the UI.
    This means that all games are processed in lockstep, and that the update rate of the UI depends on the time it takes
    to process all games.
    """

    def __init__(self, ui_class: type[UI] | None = CLI, callback_interval_frames: int = 1000) -> None:
        """Initialize the synchronous runner.

        Args:
            ui_class: The UI class to use for visualization, or None for headless mode.
            callback_interval_frames: Number of frames between interval callback invocations.
        """
        if ui_class is None:
            self._ui: UI | None = None
        elif ui_class is CLI:
            self._ui = CLI(
                randomize_background_colors_on_levelup=False,
                dynamic_background_config=DynamicLayerConfig(dynamic_background_probability=0),
            )
        else:
            self._ui = ui_class()

        self._callback_interval_frames = callback_interval_frames

    @property
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
        return {
            "ui_class": None if self._ui is None else type(self._ui),
            "callback_interval_frames": self._callback_interval_frames,
        }

    def initialize(self, num_games: int, board_size: tuple[int, int]) -> None:
        """Initialize the runner before running games.

        Args:
            num_games: Number of games that will be run.
            board_size: Tuple of (width, height) for the game boards.
        """
        if self._ui:
            self._ui.initialize(*board_size, num_boards=num_games)

    def run_games(
        self,
        games: Sequence[Game],
        max_frames: int,
        interval_callback: Callable[[], None] | None = None,
        game_over_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Run multiple games synchronously until they complete or reach max frames.

        Args:
            games: Sequence of Game instances to run.
            max_frames: Maximum number of frames to run each game.
            interval_callback: Optional callback called every callback_interval_frames.
            game_over_callback: Optional callback called when a game ends, receives game index.

        Raises:
            TypeError: If controller is not a HeuristicBotController.
            ValueError: If controller is configured to use process pool.
        """
        self._check_controller_config(games[0].controller)

        num_alive_games = len(games)

        ui_elements = UiElements(games=tuple(game.ui_elements for game in games))

        for i in range(max_frames):
            if interval_callback and i % self._callback_interval_frames == 0:
                interval_callback()

            for idx, game in enumerate(games):
                if not game.alive:
                    continue

                try:
                    game.advance_frame()
                except GameOverError:
                    num_alive_games -= 1

                    if game_over_callback:
                        game_over_callback(idx)

            if num_alive_games == 0:
                break

            if not self._ui:
                continue

            if isinstance(self._ui, CLI):
                digits = max(
                    len(f"{len(games):,}"),
                    len(f"{max_frames:,}"),
                )
                print(f"Game Over: {len(games) - num_alive_games:>{digits},} / {len(games):<{digits},}")  # noqa: T201
                print(f"Frames:    {i + 1:>{digits},} / {max_frames:<{digits},}")  # noqa: T201

            self._ui.advance_startup()
            self._ui.draw(ui_elements)

    def _check_controller_config(self, controller: Controller) -> None:
        """Validate the controller configuration.

        Args:
            controller: The controller to check.

        Raises:
            TypeError: If controller is not a HeuristicBotController.
            ValueError: If controller is configured to use process pool.
            ValueError: If controller is configured to not use lightning mode.
        """
        if not isinstance(controller, HeuristicBotController):
            msg = "SynchronousRunner requires a HeuristicBotController!"
            raise TypeError(msg)

        if controller.is_using_process_pool:
            msg = "SynchronousRunner shouldn't be used with HeuristicBotController using ProcessPoolExecutor!"
            raise ValueError(msg)

        if not controller.lightning_mode:
            msg = "SynchronousRunner should be used with HeuristicBotController using lightning mode!"
            raise ValueError(msg)
