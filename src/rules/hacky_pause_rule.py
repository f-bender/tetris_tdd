from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.clock import Clock
from game_logic.interfaces.controller import Action, Controller


class PauseRule:
    def __init__(
        self,
        controller: Controller,
        clock: Clock,
        pause_action: Action = Action(left=True, right=True, down=True),
    ) -> None:
        self._controller = controller
        self._clock = clock
        self._pause_action = pause_action

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        if self._pause_action_pressed(action_counter):
            while True:
                self._clock.tick()
                action_counter.update(self._controller.get_action())
                if self._pause_action_pressed(action_counter):
                    break

    def _pause_action_pressed(self, action_counter: ActionCounter) -> bool:
        return action_counter.held_since(self._pause_action) == 1
