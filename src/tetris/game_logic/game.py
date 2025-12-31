from tetris.exceptions import BaseTetrisError
from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.game_logic.ui_aggregator import UiAggregator


class GameOverError(BaseTetrisError):
    pass


class Game:
    def __init__(
        self,
        board: Board,
        controller: Controller,
        rule_sequence: RuleSequence | None = None,
        callback_collection: CallbackCollection | None = None,
        *,
        show_ghost_block: bool = False,
    ) -> None:
        self._board = board
        self._controller = controller
        self._action_counter = ActionCounter()
        self._rule_sequence = rule_sequence or RuleSequence.standard()
        self.callback_collection = callback_collection or CallbackCollection(())

        self._show_ghost_block = show_ghost_block

        self._frame_counter = 0
        self._alive = True
        self._game_over_frame_count = 0

        self._ui_aggregator = UiAggregator(board=board.as_array(), controller_symbol=self._controller.symbol)

        self._index = DEPENDENCY_MANAGER.current_game_index

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def game_over_frame_count(self) -> int:
        return self._game_over_frame_count

    @property
    def action_counter(self) -> ActionCounter:
        return self._action_counter

    @property
    def board(self) -> Board:
        return self._board

    @property
    def ui_elements(self) -> SingleUiElements:
        return self._ui_aggregator.ui_elements

    def reset(self) -> None:
        self._board.clear()
        self._board.active_block = None
        self._frame_counter = 0
        self._alive = True
        self._ui_aggregator.reset()

    def advance_frame(self) -> None:
        if self._frame_counter == 0:
            self.callback_collection.on_game_start(self._index)

        self._frame_counter += 1

        self._action_counter.update(self._controller.get_action())
        self.callback_collection.on_action_counter_updated(self._index)

        if self._alive:
            try:
                self._rule_sequence.apply(self._frame_counter, self._action_counter, self._board)
            except GameOverError:
                self._ui_aggregator.game_over = True
                self.callback_collection.on_game_over(self._index)
                self._alive = False
                self._game_over_frame_count = self._frame_counter
                return
            self.callback_collection.on_rules_applied(self._index)

        self._ui_aggregator.update(self._board.as_array(include_ghost=self._show_ghost_block))
