from typing import NamedTuple, override

from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.messages import (
    PostMergeFinishedMessage,
    SynchronizedSpawnCommandMessage,
    WaitingForSpawnMessage,
)


class SynchronizedSpawning(Publisher, Subscriber):
    """This class is responsible for synchronizing the spawn of pieces in the Tetris game.

    It ensures that pieces are spawned in a synchronized manner across different game instances.
    """

    def __init__(self) -> None:
        super().__init__()
        self._game_indices: set[int] = {self.game_index}
        self._game_indices_waiting_for_spawn: set[int] = set()

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.core.post_merge.post_merge_rule import PostMergeRule

        if isinstance(publisher, PostMergeRule) and publisher.game_index == self.game_index:
            return True

        if isinstance(publisher, SynchronizedSpawning) and publisher.game_index != self.game_index:
            self._game_indices.add(publisher.game_index)
            return True

        return False

    @override
    def notify(self, message: NamedTuple) -> None:
        match message:
            case PostMergeFinishedMessage():
                self._game_indices_waiting_for_spawn.add(self.game_index)
                self.notify_subscribers(WaitingForSpawnMessage(self.game_index))
            case WaitingForSpawnMessage(game_index=game_index):
                self._game_indices_waiting_for_spawn.add(game_index)
            case _:
                pass

        if len(self._game_indices_waiting_for_spawn) == len(self._game_indices):
            self._game_indices_waiting_for_spawn.clear()
            self.notify_subscribers(SynchronizedSpawnCommandMessage())
