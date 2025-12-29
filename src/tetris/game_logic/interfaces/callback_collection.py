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

    def on_game_start(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_game_start(game_index)

    def on_frame_start(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_frame_start(game_index)

    def on_action_counter_updated(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_action_counter_updated(game_index)

    def on_rules_applied(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_rules_applied(game_index)

    def on_frame_end(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_frame_end(game_index)

    def on_game_over(self, game_index: int) -> None:
        for callback in self._callbacks:
            callback.on_game_over(game_index)

    def on_all_games_over(self) -> None:
        for callback in self._callbacks:
            callback.on_all_games_over()
