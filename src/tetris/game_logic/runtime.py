from enum import Enum, auto
from itertools import count
from typing import Protocol

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.animations import ScreenHideAnimationSpec, ScreenRevealAnimationSpec
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.clock import Clock
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.dependency_manager import DependencyManager
from tetris.game_logic.interfaces.runtime_rule_sequence import RuntimeRuleSequence
from tetris.game_logic.interfaces.ui import UI, UiElements
from tetris.game_logic.sound_manager import SoundManager


class Runtime:
    def __init__(  # noqa: PLR0913
        self,
        *,
        ui: UI,
        clock: Clock,
        games: list[Game],
        controller: Controller,  # for menu navigation or startup acceleration
        rule_sequence: RuntimeRuleSequence | None = None,
        callback_collection: CallbackCollection | None = None,
        sound_manager: SoundManager | None = None,
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
        self._ui_elements = UiElements(games=tuple(game.ui_elements for game in games))

        self.state: State = STARTUP_STATE

        self._rule_sequence = rule_sequence or RuntimeRuleSequence(())
        self.callback_collection = callback_collection or CallbackCollection(())

        self._controller = controller
        self._action_counter = ActionCounter()

        self._sound_manager = sound_manager

    @property
    def action_counter(self) -> ActionCounter:
        return self._action_counter

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    @property
    def ui_elements(self) -> UiElements:
        return self._ui_elements

    @property
    def games(self) -> list[Game]:
        return self._games

    def reset(self) -> None:
        self._frame_counter = 0
        self._clock.reset()

        for game in self._games:
            game.reset()

    def run(self) -> None:
        self.callback_collection.on_runtime_start()
        while True:
            self._clock.tick()
            self.advance_frame(self._controller.get_action())

    def advance_frame(self, action: Action) -> None:
        self.callback_collection.on_frame_start(DependencyManager.RUNTIME_INDEX)

        self._action_counter.update(action)
        self.callback_collection.on_action_counter_updated(DependencyManager.RUNTIME_INDEX)

        self._rule_sequence.apply(self._frame_counter, self._action_counter)

        self.state.advance(self)

        self._ui.draw(self._ui_elements)
        self.callback_collection.on_frame_end(DependencyManager.RUNTIME_INDEX)

        self._frame_counter += 1

    def tick_is_overdue(self) -> bool:
        return self._clock.overdue()

    def advance_ui_startup(self) -> bool:
        return self._ui.advance_startup()


# State pattern


class StartupState:
    ACCELERATION_FACTOR = 0.25
    ACCELERATION_ACTION = Action(down=True)

    @classmethod
    def advance(cls, runtime: Runtime) -> None:
        for i in count():
            finished = runtime.advance_ui_startup()
            if finished:
                runtime.state = PLAYING_STATE
                break

            if runtime.tick_is_overdue() or i >= cls.ACCELERATION_FACTOR * runtime.action_counter.held_since(
                cls.ACCELERATION_ACTION
            ):
                break


PAUSE_ACTION = Action(cancel=True)


class PlayingState:
    @classmethod
    def advance(cls, runtime: Runtime) -> None:
        if runtime.action_counter.held_since(PAUSE_ACTION) == 1:
            runtime.state = PAUSED_STATE

        for game in runtime.games:
            game.advance_frame()

        if not any(game.alive for game in runtime.games):
            runtime.callback_collection.on_all_games_over()
            runtime.state = ALL_GAMES_OVER_STATE


class AllGamesOverState:
    RESTART_AFTER_GAME_OVER_ACTION = Action(confirm=True)

    class _State(Enum):
        NOT_READY = auto()
        SCREEN_HIDE = auto()
        SCREEN_REVEAL = auto()

    _STATE = _State.NOT_READY

    NUM_ACTION_FRAMES = 60
    _SCREEN_HIDE_ANIMATION = ScreenHideAnimationSpec(total_frames=NUM_ACTION_FRAMES)
    _SCREEN_REVEAL_ANIMATION = ScreenRevealAnimationSpec(total_frames=NUM_ACTION_FRAMES)

    _ready = False

    @classmethod
    def advance(cls, runtime: Runtime) -> None:
        action_held_frames = runtime.action_counter.held_since(cls.RESTART_AFTER_GAME_OVER_ACTION)

        if cls._STATE == cls._State.NOT_READY and action_held_frames == 0:
            # avoid directly starting into the middle of the animation, in case the action was already being held
            # while the games went game over (i.e. only start doing anything after the action has been released
            # once)
            cls._STATE = cls._State.SCREEN_HIDE

        if cls._STATE == cls._State.SCREEN_HIDE:
            cls._advance_screen_hide(runtime, action_held_frames=action_held_frames)

            if cls._SCREEN_HIDE_ANIMATION.done:
                for game in runtime.games:
                    game.reset()
                cls._STATE = cls._State.SCREEN_REVEAL

        if cls._STATE == cls._State.SCREEN_REVEAL:
            cls._advance_screen_reveal(runtime)

            if cls._SCREEN_REVEAL_ANIMATION.done:
                runtime.ui_elements.animations = []
                cls._SCREEN_HIDE_ANIMATION.current_frame = -1
                cls._SCREEN_REVEAL_ANIMATION.current_frame = -1
                cls._STATE = cls._State.NOT_READY
                runtime.state = PLAYING_STATE

    @classmethod
    def _advance_screen_hide(cls, runtime: Runtime, action_held_frames: int) -> None:
        if action_held_frames > 0:
            cls._SCREEN_HIDE_ANIMATION.current_frame += 1
        elif cls._SCREEN_HIDE_ANIMATION.current_frame >= 0:
            cls._SCREEN_HIDE_ANIMATION.current_frame -= 1

        if cls._SCREEN_HIDE_ANIMATION.current_frame >= 0:
            runtime.ui_elements.animations = [cls._SCREEN_HIDE_ANIMATION]

    @classmethod
    def _advance_screen_reveal(cls, runtime: Runtime) -> None:
        cls._SCREEN_REVEAL_ANIMATION.current_frame += 1

        if cls._SCREEN_REVEAL_ANIMATION.current_frame >= 0:
            runtime.ui_elements.animations = [cls._SCREEN_REVEAL_ANIMATION]


class PausedState:
    @classmethod
    def advance(cls, runtime: Runtime) -> None:
        if runtime.action_counter.held_since(PAUSE_ACTION) == 1:
            runtime.state = PLAYING_STATE


class State(Protocol):
    @classmethod
    def advance(cls, runtime: Runtime) -> None: ...


STARTUP_STATE = StartupState()
PLAYING_STATE = PlayingState()
PAUSED_STATE = PausedState()
ALL_GAMES_OVER_STATE = AllGamesOverState()
