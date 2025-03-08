from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Any

from tetris.clock.simple import SimpleClock
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.ui import UI
from tetris.heuristic_bot_gym.evaluators.evaluator import Runner
from tetris.ui.cli.ui import CLI


class ParallelRunner(Runner):
    def __init__(
        self,
        ui_class: type[UI] | None = CLI,
        callback_interval_frames: int = 1000,
        fps: int = 60,
        num_workers: int | None = None,
    ) -> None:
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
        if self._ui:
            self._ui.initialize(*board_size, num_boards=num_games)

    def run_games(
        self,
        games: Sequence[Game],
        max_frames: int,
        interval_callback: Callable[[], None] | None = None,
        game_over_callback: Callable[[int], None] | None = None,
    ) -> None:
        self._check_controller_config(games[0].controller)

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
                self._ui.draw(game.board.as_array() for game in games)

    def _check_controller_config(self, controller: Controller) -> None:
        if not isinstance(controller, HeuristicBotController):
            msg = "ParallelRunner requires a HeuristicBotController!"
            raise TypeError(msg)

        if not controller.is_using_process_pool:
            msg = "ParallelRunner should be used with HeuristicBotController using ProcessPoolExecutor!"
            raise ValueError(msg)

    def _run_game(self, game: Game, max_frames: int) -> None:
        for _ in range(max_frames):
            try:
                game.advance_frame()
            except GameOverError:
                return
