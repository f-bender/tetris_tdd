import time

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller


class BotAssistedController(Controller):
    """A controller that assists the player by providing hints for the next move."""

    _BOT_TOGGLE_ACTION = Action(confirm=True, cancel=True, down=True, left=True)
    _DEBOUNCE_TIME_S = 1

    def __init__(self, main_controller: Controller, bot_controller: HeuristicBotController) -> None:
        self._main_controller = main_controller
        self._bot_controller = bot_controller

        self._using_bot = False
        self._last_bot_toggle_time: float | None = None

    @property
    def bot_controller(self) -> HeuristicBotController:
        return self._bot_controller

    def get_action(self, board: Board | None = None) -> Action:
        main_controller_action = self._main_controller.get_action()

        self._handle_bot_toggle(main_controller_action)

        return self._bot_controller.get_action(board) if self._using_bot else main_controller_action

    def _handle_bot_toggle(self, main_controller_action: Action) -> None:
        if main_controller_action == self._BOT_TOGGLE_ACTION and (
            self._last_bot_toggle_time is None or time.time() - self._last_bot_toggle_time > self._DEBOUNCE_TIME_S
        ):
            self._using_bot = not self._using_bot
            self._last_bot_toggle_time = time.time()

    @property
    def symbol(self) -> str:
        return f"{self._main_controller.symbol}+{self._bot_controller.symbol}"
