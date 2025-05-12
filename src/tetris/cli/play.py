import contextlib
import logging
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Literal

import click

from tetris.cli.common import BoardSize
from tetris.clock.simple import SimpleClock
from tetris.controllers.bot_assisted import BotAssistedController
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.stub import StubController
from tetris.game_logic.components import Board
from tetris.game_logic.components.block import Block
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Action, Controller
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER, DependencyManager
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.game_logic.rules.core.spawn_drop_merge.synchronized_spawn import SynchronizedSpawning
from tetris.game_logic.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.game_logic.rules.monitoring.track_score_rule import ScoreTracker
from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.game_logic.runtime import Runtime
from tetris.ui.cli import CLI

if TYPE_CHECKING:
    from tetris.game_logic.interfaces.rule import Rule

LOGGER = logging.getLogger(__name__)

type ControllerParameter = Literal[
    "arrows",
    "wasd",
    "vim",
    "gamepad",
    "bot",
    "arrows+",
    "wasd+",
    "vim+",
    "gamepad+",
]


@click.command()
@click.option(
    "-n",
    "--num-games",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Number of games. If controllers are specified, it defaults to the number of specified controllers. If higher "
        "than the number of controllers, the remaining number of games is filled up with bot controllers."
    ),
)
@click.option(
    "-c",
    "--controller",
    type=click.Choice(
        [
            "arrows",
            "wasd",
            "vim",
            "gamepad",
            "bot",
            "arrows+",
            "wasd+",
            "vim+",
            "gamepad+",
        ],
        case_sensitive=False,
    ),
    multiple=True,
    help=(
        "Controller(s) to use. "
        "Case 1: --num-games is not specified: "
        "If specified, it determines the number of games and their controllers. If not specified, a single game with "
        "keyboard controller is created. "
        "Case 2: --num-games is specified: "
        "If specified once, all games will use that same controller. If specified multiple times, specifies the first "
        "N games' controllers, with the rest being filled up with bots. If not specified, all available keyboard and "
        "gamepad controllers are used, and the rest of games are filled up with bots. "
        "Note: 'gamepad' can be specified more than once if and only if more than one gamepad is connected. "
        "Add a '+' suffix to make the controller bot-assisted, i.e. allow handing over the controls to a bot by "
        "pressing down + left + confirm + cancel (simultaneously)."
    ),
)
@click.option(
    "--tetris99/--no-tetris99", default=False, show_default=True, help="Enable/disable Tetris99 attack rules."
)
@click.option(
    "--synchronize-spawning/--no-synchronize-spawning",
    default=False,
    show_default=True,
    help="Enable/disable synchronization of block spawning across games.",
)
@click.option(
    "--process-pool/--no-process-pool",
    default=True,
    show_default=True,
    help=(
        "Whether to use a process pool for Bot controllers, speeding up their planning, giving them a higher chance "
        "to react in time."
    ),
)
@click.option(
    "--fps",
    type=click.FloatRange(min=0, min_open=True),
    default=60,
    show_default=True,
    help=(
        "Target frames per second. "
        "Note that the game is designed with 60 FPS in mind and increasing FPS speeds up gameplay."
    ),
)
@click.option(
    "--board-size",
    type=BoardSize(),
    default="20x10",
    show_default=True,
    help="Height and width of the tetris board, separated by 'x' (same for all games).",
)
@click.option(
    "--block-selection",
    type=click.Choice(["truly_random", "from_shuffled_bag"]),
    default="truly_random",
    show_default=True,
    help="How to choose blocks to spawn.",
)
@click.option(
    "--seed",
    type=str,
    multiple=True,
    help=(
        "Seed for block spawning algorithm. If provided once, all games use that same seed. If provided multiple "
        "times, the number of seeds must match with the number of games (specified via --num-games of the number of "
        "--controllers), and each game gets the respective specified seed. If not provided, a random seed is "
        "used for each game."
    ),
)
def play(  # noqa: PLR0913
    *,
    num_games: int | None,
    controller: tuple[ControllerParameter, ...],
    tetris99: bool,
    synchronize_spawning: bool,
    process_pool: bool,
    fps: float,
    board_size: tuple[int, int],
    seed: tuple[str, ...],
    block_selection: Literal["truly_random", "from_shuffled_bag"],
) -> None:
    """Play Tetris with configurable rules and controllers."""
    boards, controllers = _create_boards_and_controllers(
        board_size=board_size, num_games_parameter=num_games, controllers_parameter=controller
    )
    any_bots = any(isinstance(controller, HeuristicBotController | BotAssistedController) for controller in controllers)

    seeds = _create_seeds(seed_parameters=seed, num_games=len(boards))
    block_selection_fns = [getattr(SpawnStrategyImpl, f"{block_selection}_selection_fn")(seed) for seed in seeds]

    # note: max_workers defaults to number of processors
    with ProcessPoolExecutor() if process_pool and any_bots else contextlib.nullcontext() as process_pool_or_none:
        # If process pool is used, set the process pool for all HeuristicBotControllers
        if process_pool_or_none:
            for controller_ in controllers:
                if isinstance(controller_, HeuristicBotController):
                    controller_.process_pool = process_pool_or_none
                if isinstance(controller_, BotAssistedController):
                    controller_.bot_controller.process_pool = process_pool_or_none

        games = _create_games(
            boards=boards,
            controllers=controllers,
            block_selection_fns=block_selection_fns,
            tetris99=tetris99,
            synchronize_spawning=synchronize_spawning,
        )

        runtime = _create_runtime(games, fps=fps)

        DEPENDENCY_MANAGER.wire_up(runtime=runtime, games=games)

        runtime.run()


def _create_boards_and_controllers(
    board_size: tuple[int, int],
    num_games_parameter: int | None,
    controllers_parameter: tuple[ControllerParameter, ...],
) -> tuple[list[Board], list[Controller]]:
    num_games = num_games_parameter or (len(controllers_parameter) if controllers_parameter else 1)
    if len(controllers_parameter) > num_games:
        msg = "Number of controllers cannot exceed number of games."
        raise click.BadParameter(msg)

    boards = [Board.create_empty(*board_size) for _ in range(num_games)]

    # NOTE: we avoid importing the keyboard controller unless we actually need it, to avoid unnecessary crashes on
    # systems where keyboard input is not supported (e.g. headless servers)

    if num_games_parameter is None:
        # Case 1: --num-games is not specified
        if controllers_parameter:
            # If controller is specified, it determines the number of games and their controllers.
            controllers: list[Controller] = [
                _create_controller(controller_parameter, board)
                for controller_parameter, board in zip(controllers_parameter, boards, strict=True)
            ]
        else:
            # If controller is not specified, a single game with keyboard controller is created.
            from tetris.controllers.keyboard.pynput import PynputKeyboardController

            controllers = [PynputKeyboardController()]
    else:  # noqa: PLR5501
        # Case 2: --num-games is specified
        if controllers_parameter:
            # If controller is specified once, all games will use that same controller.
            if len(controllers_parameter) == 1:
                controllers_parameter *= num_games

            # If specified multiple times, specifies the first N games' controllers, with the rest being filled up with
            # bots.
            controllers = [
                _create_controller(controller_parameter, board)
                for controller_parameter, board in zip(
                    controllers_parameter, boards[: len(controllers_parameter)], strict=True
                )
            ]
            if len(controllers) < num_games:
                controllers += [HeuristicBotController(board) for board in boards[len(controllers_parameter) :]]
        else:
            # If not specified, all available keyboard and gamepad controllers are used, and the rest of games are
            # filled up with bots.
            controllers = _default_controllers()[:num_games]
            if len(controllers) < num_games:
                controllers += [HeuristicBotController(board) for board in boards[len(controllers) :]]

    return boards, controllers


def _create_controller(controller_parameter: ControllerParameter, board: Board) -> Controller:
    if controller_parameter == "bot":
        return HeuristicBotController(board)

    bot_assisted = controller_parameter.endswith("+")

    controller: Controller
    if controller_parameter.startswith("arrows"):
        from tetris.controllers.keyboard.pynput import PynputKeyboardController

        controller = PynputKeyboardController.arrow_keys()
    elif controller_parameter.startswith("wasd"):
        from tetris.controllers.keyboard.pynput import PynputKeyboardController

        controller = PynputKeyboardController.wasd()
    elif controller_parameter.startswith("vim"):
        from tetris.controllers.keyboard.pynput import PynputKeyboardController

        controller = PynputKeyboardController.vim()
    elif controller_parameter.startswith("gamepad"):
        try:
            from inputs import devices
        except ImportError as e:
            msg = (
                "The Gamepad controller requires the extra `gamepad` dependency to be installed using "
                "`pip install tetris[gamepad]` or `uv sync --extra gamepad`!"
            )
            raise click.BadParameter(msg) from e

        from tetris.controllers.gamepad import GamepadController

        _create_controller.gamepad_index = getattr(_create_controller, "gamepad_index", -1) + 1  # type: ignore[attr-defined]
        if _create_controller.gamepad_index >= (num_gamepads := len(devices.gamepads)):  # type: ignore[attr-defined]
            msg = f"Specified more than {num_gamepads} gamepad controllers with only {num_gamepads} gamepads connected."
            raise click.BadParameter(msg)

        controller = GamepadController(gamepad_index=_create_controller.gamepad_index)  # type: ignore[attr-defined]
    else:
        msg = "Unknown controller parameter."
        raise AssertionError(msg)

    if bot_assisted:
        controller = BotAssistedController(controller, HeuristicBotController(board))

    return controller


def _default_controllers() -> list[Controller]:
    from tetris.controllers.keyboard.pynput import PynputKeyboardController

    default_controllers: list[Controller] = [PynputKeyboardController.arrow_keys()]

    with contextlib.suppress(ImportError):
        from inputs import devices

        from tetris.controllers.gamepad import GamepadController

        default_controllers.extend(GamepadController(gamepad_index=i) for i in range(len(devices.gamepads)))

    default_controllers.append(PynputKeyboardController.wasd())
    default_controllers.append(PynputKeyboardController.vim())

    return default_controllers


def _create_seeds(seed_parameters: tuple[str, ...], num_games: int) -> tuple[int, ...]:
    if not seed_parameters:
        return tuple(random.randrange(2**32) for _ in range(num_games))

    if seed_parameters == ("same",):
        seeds = (random.randrange(2**32),) * num_games
    else:
        try:
            seeds = tuple(int(s) for s in seed_parameters)
        except ValueError as e:
            msg = "Seeds must be integers."
            raise click.BadParameter(msg) from e

        if len(seeds) == 1:
            seeds *= num_games
        elif len(seeds) != num_games:
            msg = "Number of seeds must match number of games or be 1."
            raise click.BadParameter(msg)

    return seeds


def _create_games(
    boards: list[Board],
    controllers: list[Controller],
    block_selection_fns: list[Callable[[], Block]],
    *,
    tetris99: bool,
    synchronize_spawning: bool,
) -> list[Game]:
    games: list[Game] = []

    for idx, (controller, board, block_selection_fn) in enumerate(
        zip(controllers, boards, block_selection_fns, strict=True)
    ):
        DEPENDENCY_MANAGER.current_game_index = idx

        # not useless; Dependency manager will keep track of it and it will be subscribed to line clear events and
        # publish its score to the ui aggregator
        ScoreTracker()

        if isinstance(controller, Callback | Subscriber | Publisher):
            controller.game_index = idx

        if isinstance(controller, BotAssistedController):
            controller.bot_controller.game_index = idx

        rule_sequence = _create_rules_and_callbacks(
            num_games=len(boards),
            create_tetris_99_rule=tetris99,
            synchronize_spawning=synchronize_spawning,
            block_selection_fn=block_selection_fn,
        )

        games.append(Game(board=board, controller=controller, rule_sequence=rule_sequence))

    return games


def _create_rules_and_callbacks(
    num_games: int,
    *,
    create_tetris_99_rule: bool,
    synchronize_spawning: bool,
    block_selection_fn: Callable[[], Block] = Block.create_random,
) -> RuleSequence:
    """Get rules relevant for one instance of a game.

    Optionally also create (but don't return) callbacks to be added to game's/runtime's callback collection by wire-up
    function.

    Args:
        num_games: Total number of games being created overall.
        create_tetris_99_rule: Whether to create a Tetris99Rule in case there are multiple games.
        synchronize_spawning: Whether to synchronize spawning in case there are multiple games.
        block_selection_fn: Function to select blocks to spawn.

    Returns:
        A RuleSequence to be passed to the Game.
    """
    synchronize_spawning = synchronize_spawning and num_games > 1

    rules: list[Rule] = [
        MoveRule(),
        RotateRule(),
        SpawnDropMergeRule(
            spawn_strategy=SpawnStrategyImpl(block_selection_fn), synchronized_spawn=synchronize_spawning
        ),
    ]

    if create_tetris_99_rule and num_games > 1:
        rules.append(Tetris99Rule(target_idxs=list(set(range(num_games)) - {DEPENDENCY_MANAGER.current_game_index})))

    if synchronize_spawning:
        SynchronizedSpawning()  # not useless; will be added to DEPENDENCY_MANAGER.all_callbacks

    return RuleSequence(rules)


def _create_runtime(games: list[Game], *, controller: Controller | None = None, fps: float = 60000) -> Runtime:
    DEPENDENCY_MANAGER.current_game_index = DependencyManager.RUNTIME_INDEX

    TrackPerformanceCallback(fps)  # not useless; will be added to DEPENDENCY_MANAGER.all_callbacks

    if controller is not None:
        return Runtime(ui=CLI(), clock=SimpleClock(fps), games=games, controller=controller)

    try:
        from tetris.controllers.keyboard.pynput import PynputKeyboardController
    except Exception:  # noqa: BLE001
        return Runtime(ui=CLI(), clock=SimpleClock(fps), games=games, controller=StubController(Action()))

    return Runtime(ui=CLI(), clock=SimpleClock(fps), games=games, controller=PynputKeyboardController())
