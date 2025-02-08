from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tetris.game_logic.game import Game
    from tetris.game_logic.interfaces.callback import Callback
    from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
    from tetris.game_logic.runtime import Runtime


class DependencyManager:
    RUNTIME_INDEX = -1

    def __init__(self) -> None:
        self.all_subscribers: list[Subscriber] = []
        self.all_publishers: list[Publisher] = []
        self.all_callbacks: list[Callback] = []
        self.current_game_index: int = self.RUNTIME_INDEX

    def wire_up(
        self,
        runtime: "Runtime | None" = None,
        games: Iterable["Game"] | None = None,
    ) -> None:
        self._wire_up_pubs_subs()
        self._wire_up_callbacks(runtime, games)
        self.reset()

    def reset(self) -> None:
        self.all_subscribers = []
        self.all_publishers = []
        self.all_callbacks = []
        self.current_game_index = self.RUNTIME_INDEX

    def _wire_up_pubs_subs(self) -> None:
        for subscriber in self.all_subscribers:
            subscriptions: list[Publisher] = []

            for publisher in self.all_publishers:
                if subscriber.should_be_subscribed_to(publisher):
                    publisher.add_subscriber(subscriber)
                    subscriptions.append(publisher)

            subscriber.verify_subscriptions(subscriptions)

    def _wire_up_callbacks(self, runtime: "Runtime | None" = None, games: Iterable["Game"] | None = None) -> None:
        from tetris.game_logic.interfaces.callback_collection import CallbackCollection

        if runtime is not None:
            runtime.callback_collection = CallbackCollection(  # type: ignore[has-type]
                tuple(callback for callback in self.all_callbacks if callback.should_be_called_by(self.RUNTIME_INDEX))
            )

        if games is not None:
            for idx, game in enumerate(games):
                game.callback_collection = CallbackCollection(
                    tuple(callback for callback in self.all_callbacks if callback.should_be_called_by(idx))
                )


DEPENDENCY_MANAGER = DependencyManager()
