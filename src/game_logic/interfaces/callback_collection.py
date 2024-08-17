from collections.abc import Collection
from typing import NamedTuple

from game_logic.interfaces.callback import Callback


# Note that `CallbackCollection` itself implements the `Collection` Protocol, so it could be seen as an implementation
# of the Composite pattern
class CallbackCollection:
    def __init__(self, callbacks: Collection[Callback]) -> None:
        self._callbacks = callbacks

    def custom_message(self, message: NamedTuple) -> None:
        for callback in self._callbacks:
            callback.custom_message(message)

    def on_game_start(self) -> None:
        for callback in self._callbacks:
            callback.on_game_start()

    def on_frame_start(self) -> None:
        for callback in self._callbacks:
            callback.on_frame_start()

    def on_action_counter_updated(self) -> None:
        for callback in self._callbacks:
            callback.on_action_counter_updated()

    def on_rules_applied(self) -> None:
        for callback in self._callbacks:
            callback.on_rules_applied()

    def on_frame_end(self) -> None:
        for callback in self._callbacks:
            callback.on_frame_end()
