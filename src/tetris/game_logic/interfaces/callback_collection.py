from collections.abc import Collection

from tetris.game_logic.interfaces.callback import Callback


# Note that `CallbackCollection` itself implements the `Collection` Protocol, so it could be seen as an implementation
# of the Composite pattern
class CallbackCollection:
    def __init__(self, callbacks: Collection[Callback]) -> None:
        self._callbacks = callbacks

    def on_runtime_start(self) -> None:
        for callback in self._callbacks:
            callback.on_runtime_start()

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

    def on_game_over(self) -> None:
        for callback in self._callbacks:
            callback.on_game_over()
