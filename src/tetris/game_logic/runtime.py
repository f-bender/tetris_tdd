from itertools import count
from typing import Protocol

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.clock import Clock
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.ui import UI


class Runtime:
    def __init__(
        self,
        ui: UI,
        clock: Clock,
        games: list[Game],
        controller: Controller,  # for menu navigation or startup acceleration
        callback_collection: CallbackCollection | None = None,
    ) -> None:
        if not games:
            msg = "At least one game must be provided"
            raise ValueError(msg)

        board_size = games[0].board.size
        if not all(game.board.size == board_size for game in games[1:]):
            msg = "All games must have the same board size"
            raise ValueError(msg)

        self._games = games
        self._ui = ui
        self._ui.initialize(*board_size, num_boards=len(games))
        self._clock = clock
        self._frame_counter = 0
        self._state: State = STARTUP_STATE

        self._callback_collection = callback_collection or CallbackCollection(())

        self._controller = controller
        self._action_counter = ActionCounter()

    @property
    def state(self) -> "State":
        return self._state

    @state.setter
    def state(self, state: "State") -> None:
        self._state = state

    @property
    def action_counter(self) -> ActionCounter:
        return self._action_counter

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def games(self) -> list[Game]:
        return self._games

    def reset(self) -> None:
        self._frame_counter = 0
        self._clock.reset()
        for game in self._games:
            game.reset()

    def run(self) -> None:
        self._callback_collection.on_runtime_start()
        while True:
            self._clock.tick()
            self.advance_frame(self._controller.get_action())

    def advance_frame(self, action: Action) -> None:
        self._callback_collection.on_frame_start()

        self._action_counter.update(action)
        self._callback_collection.on_action_counter_updated()

        self._state.advance(self)

        self._ui.draw(game.board.as_array() for game in self._games)
        self._callback_collection.on_frame_end()

        self._frame_counter += 1

    def tick_is_overdue(self) -> bool:
        return self._clock.overdue()

    def advance_ui_startup(self) -> bool:
        return self._ui.advance_startup()


# State pattern


class StartupState:
    ACCELERATION_FACTOR = 0.1
    ACCELERATION_ACTION = Action(down=True)

    def advance(self, runtime: Runtime) -> None:
        for i in count():
            finished = runtime.advance_ui_startup()
            if finished:
                runtime.state = PLAYING_STATE
                break

            if runtime.tick_is_overdue() or i >= self.ACCELERATION_FACTOR * runtime.action_counter.held_since(
                self.ACCELERATION_ACTION
            ):
                break


PAUSE_ACTION = Action(cancel=True)


class PlayingState:
    def advance(self, runtime: Runtime) -> None:
        if runtime.action_counter.held_since(PAUSE_ACTION) == 1:
            runtime.state = PAUSED_STATE

        for game in runtime.games:
            try:
                game.advance_frame()
            except GameOverError:
                game.reset()


class PausedState:
    def advance(self, runtime: Runtime) -> None:
        if runtime.action_counter.held_since(PAUSE_ACTION) == 1:
            runtime.state = PLAYING_STATE


class State(Protocol):
    def advance(self, runtime: Runtime) -> None: ...


STARTUP_STATE = StartupState()
PLAYING_STATE = PlayingState()
PAUSED_STATE = PausedState()
