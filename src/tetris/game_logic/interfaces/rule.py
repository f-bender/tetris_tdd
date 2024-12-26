from typing import TYPE_CHECKING, Protocol

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection

if TYPE_CHECKING:
    from tetris.game_logic.game import GameState


class Rule(Protocol):
    def apply(
        self,
        frame_counter: int,
        action_counter: ActionCounter,
        board: Board,
        callback_collection: CallbackCollection,
        state: "GameState",
    ) -> None: ...
