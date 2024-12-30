from abc import ABC, abstractmethod
from typing import NamedTuple, Protocol

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection


class Rule(Protocol):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
        callback_collection: CallbackCollection,
    ) -> None: ...


# Publisher-Subscriber pattern for cross-rule communication
class Publisher:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def add_subscriber(self, subscriber: "Subscriber") -> None:
        self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: "Subscriber") -> None:
        self._subscribers.remove(subscriber)

    def notify_subscribers(self, message: NamedTuple) -> None:
        for subscriber in self._subscribers:
            subscriber.notify(message)


class Subscriber(ABC):
    @abstractmethod
    def notify(self, message: NamedTuple) -> None: ...
