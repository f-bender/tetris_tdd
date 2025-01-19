"""Publisher-Subscriber pattern for communication between e.g. rules."""

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

from tetris.game_logic.interfaces import global_current_game_index

LOGGER = logging.getLogger(__name__)


class Subscriber(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.game_index = global_current_game_index.current_game_index
        ALL_SUBSCRIBERS.append(self)

    @abstractmethod
    def notify(self, message: NamedTuple) -> None: ...

    @abstractmethod
    def should_be_subscribed_to(self, publisher: "Publisher") -> bool: ...

    def verify_subscriptions(self, publishers: list["Publisher"]) -> None:
        if not publishers:
            LOGGER.warning("%s has no subscriptions!", self)


class Publisher:
    def __init__(self) -> None:
        super().__init__()

        self._subscribers: list[Subscriber] = []

        self.game_index = global_current_game_index.current_game_index
        ALL_PUBLISHERS.append(self)

    def add_subscriber(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        self._subscribers.remove(subscriber)

    def notify_subscribers(self, message: NamedTuple) -> None:
        for subscriber in self._subscribers:
            subscriber.notify(message)


ALL_SUBSCRIBERS: list[Subscriber] = []
ALL_PUBLISHERS: list[Publisher] = []
