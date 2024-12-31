import contextlib
from typing import TYPE_CHECKING

from tetris.clock.simple import SimpleClock
from tetris.controllers.gamepad import GamepadController
from tetris.controllers.keyboard import KeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.runtime import Runtime
from tetris.logging_config import configure_logging
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.rules.special.parry_rule import ParryRule
from tetris.ui.cli import CLI

if TYPE_CHECKING:
    from tetris.game_logic.interfaces.rule import Rule

FPS = 60


def main() -> None:
    configure_logging()

    games, callbacks = create_games(num_games=3)
    runtime = create_runtime(games, callbacks)

    runtime.run()


def create_games(
    num_games: int = 1,
    controllers: list[Controller] | None = None,
    names: list[str] | None = None,
    board_size: tuple[int, int] = (20, 10),
) -> tuple[list[Game], list[Callback]]:
    if controllers is not None and len(controllers) != num_games:
        msg = f"Number of controllers ({len(controllers)}) doesn't match number of games ({num_games})!"
        raise ValueError(msg)

    if names is not None and len(names) != num_games:
        msg = f"Number of names ({len(names)}) doesn't match number of games ({num_games})!"
        raise ValueError(msg)

    if controllers is None:
        default_controllers = _default_controllers()

        if len(default_controllers) < num_games:
            msg = f"Not enough controllers available: {len(default_controllers)} controllers, {num_games} games"
            raise ValueError(msg)

        controllers = default_controllers[:num_games]

    if names is None:
        names = [f"Player {i}" for i in range(1, num_games + 1)]

    tetris_99_rules = [Tetris99Rule(id=i, target_ids=list(set(range(num_games)) - {i})) for i in range(num_games)]

    for publisher in tetris_99_rules:
        for observer in tetris_99_rules:
            if publisher is not observer:
                publisher.add_subscriber(observer)

    runtime_callbacks: list[Callback] = []

    games: list[Game] = []
    for controller, name, tetris_99_rule in zip(controllers, names, tetris_99_rules, strict=True):
        track_score_callback = TrackScoreCallback(header=name)
        runtime_callbacks.append(track_score_callback)

        rule_sequence, callback_collection = _create_rules_and_callbacks(tetris_99_rule, track_score_callback)
        games.append(
            Game(
                board=Board.create_empty(*board_size),
                controller=controller,
                rule_sequence=rule_sequence,
                callback_collection=callback_collection,
            )
        )

    return games, runtime_callbacks


def _default_controllers() -> list[Controller]:
    default_controllers: list[Controller] = [KeyboardController.arrow_keys()]

    with contextlib.suppress(ImportError):
        from inputs import devices

        default_controllers.extend(GamepadController(gamepad_index=i) for i in range(len(devices.gamepads)))

    default_controllers.append(KeyboardController.wasd())
    default_controllers.append(KeyboardController.vim())

    return default_controllers


def create_runtime(games: list[Game], callbacks: list[Callback] | None = None, fps: float = FPS) -> Runtime:
    ui = CLI()
    clock = SimpleClock(fps=fps)
    return Runtime(
        ui,
        clock,
        games,
        KeyboardController(),
        callback_collection=CallbackCollection((TrackPerformanceCallback(fps=fps), *(callbacks or []))),
    )


def _create_rules_and_callbacks(
    tetris99_rule: Tetris99Rule | None = None, track_score_callback: TrackScoreCallback | None = None
) -> tuple[RuleSequence, CallbackCollection]:
    """Get rules and callbacks relevant for one instance of a game.

    Returns:
        - a RuleSequence to be passed to the Game
        - a CallbackCollection to be passed to the Game
    """
    spawn_drop_merge_rule = SpawnDropMergeRule()
    parry_rule = ParryRule(leeway_frames=1)
    clear_full_lines_rule = ClearFullLinesRule()

    if tetris99_rule:
        clear_full_lines_rule.add_subscriber(tetris99_rule)

    if track_score_callback:
        clear_full_lines_rule.add_subscriber(track_score_callback)

    spawn_drop_merge_rule.add_subscriber(parry_rule)

    rules: list[Rule] = [
        MoveRule(),
        RotateRule(),
        spawn_drop_merge_rule,
        parry_rule,
        clear_full_lines_rule,
    ]
    if tetris99_rule:
        rules.append(tetris99_rule)
    rule_sequence = RuleSequence(rules)

    callbacks: list[Callback] = [spawn_drop_merge_rule]
    if track_score_callback:
        callbacks.append(track_score_callback)
    callback_collection = CallbackCollection(callbacks)

    return rule_sequence, callback_collection


if __name__ == "__main__":
    main()
