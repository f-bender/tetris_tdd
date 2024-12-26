from itertools import count

from tetris.exceptions import BaseTetrisError
from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components import Board
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.clock import Clock
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import UI


class GameOverError(BaseTetrisError):
    pass


class Game:
    def __init__(  # noqa: PLR0913
        self,
        ui: UI,
        board: Board,
        controller: Controller,
        clock: Clock,
        rule_sequence: RuleSequence,
        callback_collection: CallbackCollection | None = None,
    ) -> None:
        self._ui = ui
        self._board = board
        self._ui.initialize(self._board.height, self._board.width)
        self._controller = controller
        self._clock = clock
        self._frame_counter = 0
        self._action_counter = ActionCounter()
        self._rule_sequence = rule_sequence
        self._callback_collection = callback_collection or CallbackCollection(())
        self._startup = True

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    def reset(self) -> None:
        self._frame_counter = 0
        self._board.clear()
        self._clock.reset()

    def run(self) -> None:
        self._callback_collection.on_game_start()
        while True:
            self._clock.tick()
            try:
                self.advance_frame(self._controller.get_action())
            except GameOverError:
                return

    MAX_STARTUP_STEPS_PER_FRAME = 5

    def advance_frame(self, action: Action) -> None:
        self._callback_collection.on_frame_start()

        self._action_counter.update(action)
        self._callback_collection.on_action_counter_updated()

        if self._startup:
            for i in count():
                finished = self._ui.advance_startup()
                if finished:
                    self._startup = False
                    break

                if (
                    self._action_counter.held_since(Action(down=True)) == 0
                    or self._clock.overdue()
                    or i >= self.MAX_STARTUP_STEPS_PER_FRAME
                ):
                    break
        else:
            self._rule_sequence.apply(self._frame_counter, self._action_counter, self._board, self._callback_collection)
            self._callback_collection.on_rules_applied()

        self._ui.draw(self._board.as_array())
        self._callback_collection.on_frame_end()

        self._frame_counter += 1
