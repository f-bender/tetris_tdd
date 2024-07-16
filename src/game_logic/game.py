from exceptions import BaseTetrisException

from game_logic.action_counter import ActionCounter
from game_logic.components import Board
from game_logic.interfaces.clock import Clock
from game_logic.interfaces.controller import Action, Controller
from game_logic.interfaces.rule_sequence import RuleSequence
from game_logic.interfaces.ui import UI


class GameOver(BaseTetrisException):
    pass


class Game:
    def __init__(self, ui: UI, board: Board, controller: Controller, clock: Clock, rule_sequence: RuleSequence) -> None:
        self._ui = ui
        self._board = board
        self._controller = controller
        self._clock = clock
        self._frame_counter = 0
        self._action_counter = ActionCounter()
        self._rule_sequence = rule_sequence

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    def run(self) -> None:
        self.initialize()
        while True:
            self._clock.tick()
            try:
                self.advance_frame(self._controller.get_action())
            except GameOver:
                self._ui.game_over(self._board.as_array())
                return

    def initialize(self) -> None:
        self._ui.initialize(self._board.height, self._board.width)

    def advance_frame(self, action: Action) -> None:
        self._action_counter.update(action)
        self._rule_sequence.apply(self._frame_counter, self._action_counter, self._board)
        self._ui.draw(self._board.as_array())
        self._frame_counter += 1
