import time
from typing import NamedTuple, override

from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.messages import BotAssistanceEnd, BotAssistanceStart, ControllerSymbolUpdatedMessage
from tetris.game_logic.rules.special.powerup_effect import BotAssistanceEffect


class BotAssistedController(Subscriber, Publisher, Callback, Controller):
    """A controller that assists the player by providing hints for the next move."""

    _BOT_TOGGLE_ACTION = Action(confirm=True, cancel=True, down=True, left=True)
    _DEBOUNCE_TIME_S = 1

    def __init__(
        self,
        main_controller: Controller,
        bot_controller: HeuristicBotController,
        *,
        allow_manual_activation: bool = False,
    ) -> None:
        super().__init__()

        self._main_controller = main_controller
        self._bot_controller = bot_controller
        self._allow_manual_activation = allow_manual_activation

        self._using_bot = False
        self._last_bot_toggle_time: float | None = None

    @override
    def on_game_start(self) -> None:
        self._using_bot = False
        self._last_bot_toggle_time = None

    @property
    def bot_controller(self) -> HeuristicBotController:
        return self._bot_controller

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return (
            not self._allow_manual_activation
            and isinstance(publisher, BotAssistanceEffect)
            and publisher.game_index == self.game_index
        )

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if not self._allow_manual_activation and not any(
            isinstance(publisher, BotAssistanceEffect) for publisher in publishers
        ):
            msg = (
                f"{type(self).__name__} of game {self.game_index} is not manually toggleable, and not subscribed to a "
                f"BotAssistanceEffect: {publishers}"
            )
            raise RuntimeError(msg)

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case BotAssistanceStart():
                self._using_bot = True
                self.notify_subscribers(ControllerSymbolUpdatedMessage(self.symbol))
            case BotAssistanceEnd():
                self._using_bot = False
                self.notify_subscribers(ControllerSymbolUpdatedMessage(self.symbol))
            case _:
                msg = f"Unexpected message: {message}"
                raise ValueError(msg)

    @override
    def get_action(self) -> Action:
        main_controller_action = self._main_controller.get_action()

        self._handle_bot_toggle(main_controller_action)

        return self._bot_controller.get_action() if self._using_bot else main_controller_action

    def _handle_bot_toggle(self, main_controller_action: Action) -> None:
        if not self._allow_manual_activation:
            return

        if main_controller_action == self._BOT_TOGGLE_ACTION and (
            self._last_bot_toggle_time is None or time.time() - self._last_bot_toggle_time > self._DEBOUNCE_TIME_S
        ):
            self._using_bot = not self._using_bot
            self._last_bot_toggle_time = time.time()

    @property
    @override
    def symbol(self) -> str:
        if self._allow_manual_activation:
            return f"{self._main_controller.symbol}+{self._bot_controller.symbol}"

        if self._using_bot:
            return self._bot_controller.symbol

        return self._main_controller.symbol
