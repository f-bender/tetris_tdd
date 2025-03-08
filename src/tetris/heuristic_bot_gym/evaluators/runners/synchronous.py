from collections.abc import Callable, Sequence
from typing import Any

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.ui import UI
from tetris.heuristic_bot_gym.evaluators.evaluator import Runner
from tetris.ui.cli.ui import CLI


class SynchronousRunner(Runner):
    def __init__(self, ui_class: type[UI] | None = CLI, callback_interval_frames: int = 1000) -> None:
        self._ui = None if ui_class is None else ui_class()
        self._callback_interval_frames = callback_interval_frames

    @property
    def config(self) -> dict[str, Any]:
        """Config dict that can be used to create an equivalent instance of this class using cls(**config)."""
        return {
            "ui_class": None if self._ui is None else type(self._ui),
            "callback_interval_frames": self._callback_interval_frames,
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

        num_alive_games = len(games)

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
            self._ui.draw(game.board.as_array() for game in games)

    def _check_controller_config(self, controller: Controller) -> None:
        if not isinstance(controller, HeuristicBotController):
            msg = "SynchronousRunner requires a HeuristicBotController!"
            raise TypeError(msg)

        if controller.is_using_process_pool:
            msg = "SynchronousRunner shouldn't be used with HeuristicBotController using ProcessPoolExecutor!"
            raise ValueError(msg)
