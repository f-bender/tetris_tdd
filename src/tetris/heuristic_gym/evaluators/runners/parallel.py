from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Any

from tetris.clock.simple import SimpleClock
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.ui.cli.ui import CLI


class ParallelRunner:
    """A parallel runner for evaluating Tetris games.

    This runner executes games concurrently using threads. It's designed to work with HeuristicBotController in
    lightning mode, and in process-pool mode.
    Each game runs in its own thread, and the runner will update the UI periodically, as specified via `fps`.
    While these threads are not truly parallel due to the GIL, they enable multiple games to reach their hot spot at
    the same time, which are then processed truly in parallel by the controllers' shared process pool that is used at
    that hot spot.
    """

    def __init__(
        self,
        ui_class: type[UI] | None = CLI,
        callback_interval_frames: int = 1000,
        fps: int = 60,
        num_workers: int | None = None,
    ) -> None:
        """Initialize the parallel runner.

        Args:
            ui_class: The UI class to use for visualization, or None for headless mode.
            callback_interval_frames: Number of frames between interval callback invocations.
            fps: Frames per second for UI updates.
            num_workers: Number of process pool workers, defaults to the number of CPU cores if None.
        """
        self._ui = None if ui_class is None else ui_class()
        self._callback_interval_frames = callback_interval_frames
        self._fps = fps
        self.num_workers = num_workers

    @property
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
        return {
            "ui_class": None if self._ui is None else type(self._ui),
            "callback_interval_frames": self._callback_interval_frames,
            "fps": self._fps,
            "num_workers": self.num_workers,
        }

    def initialize(self, num_games: int, board_size: tuple[int, int]) -> None:
        """Initialize the runner before running games.

        Args:
            num_games: Number of games that will be run.
            board_size: Tuple of (height, width) for the game boards.
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
        """Run multiple games in parallel until they complete or reach max frames.

        Args:
            games: Sequence of Game instances to run.
            max_frames: Maximum number of frames to run each game.
            interval_callback: Optional callback called every callback_interval_frames.
            game_over_callback: Optional callback called when a game ends, receives game index.

        Raises:
            TypeError: If controller is not a HeuristicBotController.
            ValueError: If controller is not configured to use process pool.
        """
        self._check_controller_config(games[0].controller)

        ui_elements = UiElements(games=tuple(game.ui_elements for game in games))

        with ThreadPoolExecutor(max_workers=len(games)) as thread_pool:
            futures = {
                i: thread_pool.submit(self._run_game, game=games[i], max_frames=max_frames) for i in range(len(games))
            }

            clock = SimpleClock(fps=self._fps)
            for i in count():
                clock.tick()

                if interval_callback and i % self._callback_interval_frames == 0:
                    interval_callback()

                done_idxs = [idx for idx, fut in futures.items() if fut.done()]
                for done_idx in done_idxs:
                    del futures[done_idx]

                    if game_over_callback:
                        game_over_callback(done_idx)

                if not futures:
                    break

                if not self._ui:
                    continue

                if isinstance(self._ui, CLI):
                    digits = len(f"{len(games):,}")
                    print(f"Game Over: {len(games) - len(futures):>{digits},} / {len(games):<{digits},}")  # noqa: T201

                self._ui.advance_startup()
                self._ui.draw(ui_elements)

    def _run_game(self, game: Game, max_frames: int) -> None:
        """Run a single game until completion or max frames reached.

        Args:
            game: The game instance to run.
            max_frames: Maximum number of frames to run the game.
        """
        for _ in range(max_frames):
            try:
                game.advance_frame()
            except GameOverError:
                return

    def _check_controller_config(self, controller: Controller) -> None:
        """Validate the controller configuration.

        Args:
            controller: The controller to check.

        Raises:
            TypeError: If controller is not a HeuristicBotController.
            ValueError: If controller is not configured to use process pool.
            ValueError: If controller is configured to not use lightning mode.
        """
        if not isinstance(controller, HeuristicBotController):
            msg = "ParallelRunner requires a HeuristicBotController!"
            raise TypeError(msg)

        if not controller.is_using_process_pool:
            msg = "ParallelRunner should be used with HeuristicBotController using ProcessPoolExecutor!"
            raise ValueError(msg)

        if not controller.lightning_mode:
            msg = "SynchronousRunner should be used with HeuristicBotController using lightning mode!"
            raise ValueError(msg)
