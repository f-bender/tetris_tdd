import contextlib
from typing import TYPE_CHECKING

from tetris.clock.simple import SimpleClock
from tetris.controllers.gamepad import GamepadController
from tetris.controllers.heuristic_bot import HeuristicBotController
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
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import LineClearSpeedupStrategy
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

    boards = create_boards()
    controller = HeuristicBotController(boards[0], fps=FPS)
    games, callbacks = create_games(boards=boards, controllers=[controller])
    runtime = create_runtime(games, callbacks)

    runtime.run()


def create_boards(num_boards: int = 1, size: tuple[int, int] = (20, 10)) -> list[Board]:
    return [Board.create_empty(*size) for _ in range(num_boards)]


def create_games(
    num_games: int | None = None,
    *,
    controllers: list[Controller] | None = None,
    names: list[str] | None = None,
    boards: list[Board] | None = None,
) -> tuple[list[Game], list[Callback]]:
    lengths = {
        length
        for length in (
            num_games,
            len(controllers) if controllers is not None else None,
            len(names) if names is not None else None,
            len(boards) if boards is not None else None,
        )
        if length is not None
    }
    if len(lengths) > 1:
        msg = "Mismatched lengths of arguments: num_games, controllers, names, boards"
        raise ValueError(msg)

    num_games = 1 if len(lengths) == 0 else next(iter(lengths))

    if controllers is None:
        default_controllers = _default_controllers()

        if len(default_controllers) < num_games:
            msg = f"Not enough controllers available: {len(default_controllers)} controllers, {num_games} games"
            raise ValueError(msg)

        controllers = default_controllers[:num_games]

    if names is None:
        names = [f"Player {i}" for i in range(1, num_games + 1)]

    boards = boards or create_boards(num_games)

    tetris_99_rules: list[Tetris99Rule] | list[None] = [None] if num_games == 1 else _create_tetris_99_rules(num_games)

    runtime_callbacks: list[Callback] = []

    games: list[Game] = []
    for controller, name, tetris_99_rule, board in zip(controllers, names, tetris_99_rules, boards, strict=True):
        track_score_callback = TrackScoreCallback(header=name)
        runtime_callbacks.append(track_score_callback)

        rule_sequence, callback_collection = _create_rules_and_callbacks(
            controller, tetris_99_rule, track_score_callback
        )
        games.append(
            Game(
                board=board,
                controller=controller,
                rule_sequence=rule_sequence,
                callback_collection=callback_collection,
            )
        )

    return games, runtime_callbacks


def _create_tetris_99_rules(num_games: int) -> list[Tetris99Rule]:
    if not num_games > 1:
        msg = "Tetris 99 rules require at least 2 players."
        raise ValueError(msg)

    tetris_99_rules = [Tetris99Rule(id=i, target_ids=list(set(range(num_games)) - {i})) for i in range(num_games)]

    for publisher in tetris_99_rules:
        for observer in tetris_99_rules:
            if publisher is not observer:
                publisher.add_subscriber(observer)

    return tetris_99_rules


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
    x, tetris99_rule: Tetris99Rule | None = None, track_score_callback: TrackScoreCallback | None = None
) -> tuple[RuleSequence, CallbackCollection]:
    """Get rules and callbacks relevant for one instance of a game.

    Returns:
        - a RuleSequence to be passed to the Game
        - a CallbackCollection to be passed to the Game
    """
    spawn_drop_merge_rule = SpawnDropMergeRule(
        speed_strategy=(line_clear_speedup_strategy := LineClearSpeedupStrategy()),
        spawn_strategy=(spawn_strategy := SpawnStrategyImpl()),
    )
    spawn_strategy.add_subscriber(x)
    parry_rule = ParryRule(leeway_frames=1)
    clear_full_lines_rule = ClearFullLinesRule()

    if tetris99_rule:
        clear_full_lines_rule.add_subscriber(tetris99_rule)

    if track_score_callback:
        clear_full_lines_rule.add_subscriber(track_score_callback)

    clear_full_lines_rule.add_subscriber(line_clear_speedup_strategy)
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

    callbacks: list[Callback] = [spawn_drop_merge_rule, line_clear_speedup_strategy]
    if track_score_callback:
        callbacks.append(track_score_callback)
    callback_collection = CallbackCollection(callbacks)

    return rule_sequence, callback_collection


if __name__ == "__main__":
    main()
