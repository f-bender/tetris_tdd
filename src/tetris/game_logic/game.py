from tetris.exceptions import BaseTetrisError
from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.rule_sequence import RuleSequence


class GameOverError(BaseTetrisError):
    pass


class Game:
    def __init__(
        self,
        board: Board,
        controller: Controller,
        rule_sequence: RuleSequence,
        callback_collection: CallbackCollection | None = None,
    ) -> None:
        self._board = board
        self._controller = controller
        self._action_counter = ActionCounter()
        self._rule_sequence = rule_sequence
        self.callback_collection = callback_collection or CallbackCollection(())

        self._frame_counter = 0
        self.callback_collection.on_game_start()

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def action_counter(self) -> ActionCounter:
        return self._action_counter

    @property
    def board(self) -> Board:
        return self._board

    def reset(self) -> None:
        self._board.clear()
        self._frame_counter = 0
        self.callback_collection.on_game_start()

    def advance_frame(self) -> None:
        self._frame_counter += 1

        self._action_counter.update(self._controller.get_action(self._board))
        self.callback_collection.on_action_counter_updated()

        self._rule_sequence.apply(self._frame_counter, self._action_counter, self._board)
        self.callback_collection.on_rules_applied()
