from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.messages import (
    FinishedMergeMessage,
    SynchronizedSpawnCommandMessage,
    WaitingForSpawnMessage,
)


class SynchronizedSpawning(Publisher, Subscriber, Callback):
    """This class is responsible for synchronizing the spawn of pieces in the Tetris game.

    It ensures that pieces are spawned in a synchronized manner across different game instances.
    """

    def __init__(self) -> None:
        super().__init__()
        self._game_indices: set[int] = {self.game_index}
        self._game_indices_waiting_for_spawn: set[int] = set()

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule

        if isinstance(publisher, SpawnDropMergeRule) and publisher.game_index == self.game_index:
            return True

        if isinstance(publisher, SynchronizedSpawning) and publisher.game_index != self.game_index:
            self._game_indices.add(publisher.game_index)
            return True

        return False

    def notify(self, message: NamedTuple) -> None:
        match message:
            case FinishedMergeMessage():
                self._game_indices_waiting_for_spawn.add(self.game_index)
                self.notify_subscribers(WaitingForSpawnMessage(self.game_index))
            case WaitingForSpawnMessage(game_index=game_index):
                self._game_indices_waiting_for_spawn.add(game_index)
            case _:
                pass

        if len(self._game_indices_waiting_for_spawn) == len(self._game_indices):
            self._game_indices_waiting_for_spawn.clear()
            self.notify_subscribers(SynchronizedSpawnCommandMessage())

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index
