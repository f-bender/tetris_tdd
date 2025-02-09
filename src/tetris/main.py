import contextlib
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from tetris.clock.simple import SimpleClock
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.keyboard.pynput import PynputKeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER, DependencyManager
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.runtime import Runtime
from tetris.logging_config import configure_logging
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.rules.special.parry_rule import ParryRule
from tetris.ui.cli import CLI

if TYPE_CHECKING:
    from tetris.game_logic.interfaces.rule import Rule


def main() -> None:
    configure_logging()

    boards = _create_boards(1)
    # note: max_workers defaults to number of processors
    with ProcessPoolExecutor() as process_pool:
        games = _create_games(
            boards=boards,
            controllers=[
                HeuristicBotController(board, lightning_mode=True, process_pool=process_pool) for board in boards
            ],
            use_tetris_99_rules=False,
        )
        runtime = _create_runtime(games)

        DEPENDENCY_MANAGER.wire_up(runtime=runtime, games=games)

        runtime.run()


def _create_games(
    num_games: int | None = None,
    *,
    controllers: list[Controller] | None = None,
    names: list[str] | None = None,
    boards: list[Board] | None = None,
    use_tetris_99_rules: bool = True,
) -> list[Game]:
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

    boards = boards or _create_boards(num_games)

    games: list[Game] = []

    for idx, (controller, name, board) in enumerate(zip(controllers, names, boards, strict=True)):
        DEPENDENCY_MANAGER.current_game_index = idx

        # not useless; will be added to ALL_CALLBACKS and eventually to runtime's callback collection
        TrackScoreCallback(header=name)

        if isinstance(controller, Callback | Subscriber | Publisher):
            controller.game_index = idx

        rule_sequence = _create_rules_and_callbacks(num_games=num_games, create_tetris_99_rule=use_tetris_99_rules)

        games.append(Game(board=board, controller=controller, rule_sequence=rule_sequence))

    return games


def _default_controllers() -> list[Controller]:
    default_controllers: list[Controller] = [PynputKeyboardController.arrow_keys()]

    with contextlib.suppress(ImportError):
        from inputs import devices

        from tetris.controllers.gamepad import GamepadController

        default_controllers.extend(GamepadController(gamepad_index=i) for i in range(len(devices.gamepads)))

    default_controllers.append(PynputKeyboardController.wasd())
    default_controllers.append(PynputKeyboardController.vim())

    return default_controllers


def _create_boards(num_boards: int = 1, size: tuple[int, int] = (20, 10)) -> list[Board]:
    return [Board.create_empty(*size) for _ in range(num_boards)]


def _create_rules_and_callbacks(num_games: int, *, create_tetris_99_rule: bool = True) -> RuleSequence:
    """Get rules relevant for one instance of a game.

    Optionally also create (but don't return) callbacks to be added to game's/runtime's callback collection by wire-up
    function.

    Args:
        num_games: Total number of games being created overall.
        create_tetris_99_rule: Whether to create a Tetris99Rule in case there are multiple games.

    Returns:
        A RuleSequence to be passed to the Game.
    """
    rules: list[Rule] = [
        MoveRule(),
        RotateRule(),
        SpawnDropMergeRule(spawn_delay=0),
        ParryRule(),
        ClearFullLinesRule(),
    ]

    if create_tetris_99_rule and num_games > 1:
        rules.append(Tetris99Rule(target_idxs=list(set(range(num_games)) - {DEPENDENCY_MANAGER.current_game_index})))

    return RuleSequence(rules)


def _create_runtime(games: list[Game], *, controller: Controller | None = None, fps: float = 60000) -> Runtime:
    DEPENDENCY_MANAGER.current_game_index = DependencyManager.RUNTIME_INDEX

    TrackPerformanceCallback(fps)  # not useless; will be added to DEPENDENCY_MANAGER.all_callbacks

    return Runtime(ui=CLI(), clock=SimpleClock(fps), games=games, controller=controller or PynputKeyboardController())


if __name__ == "__main__":
    main()
