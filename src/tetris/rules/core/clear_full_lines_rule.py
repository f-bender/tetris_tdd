from typing import NamedTuple

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.rules.core.messages import InstantSpawnMessage, LineClearMessage


class ClearFullLinesRule(Publisher, Subscriber):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
    ) -> None:
        self._clear_full_lines(board)

    def _clear_full_lines(self, board: Board) -> None:
        full_lines = board.get_full_line_idxs()
        if not full_lines:
            return

        board.clear_lines(full_lines)
        message = LineClearMessage(cleared_lines=full_lines)
        self.notify_subscribers(message)

    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, InstantSpawnMessage):
            self._clear_full_lines(message.board)

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        # Avoid circular import
        from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule

        return isinstance(publisher, SpawnDropMergeRule) and publisher.game_index == self.game_index
