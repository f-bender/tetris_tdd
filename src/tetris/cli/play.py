import contextlib
import random
from collections.abc import Callable, Collection
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
from tetris.game_logic.interfaces.runtime_rule_sequence import RuntimeRuleSequence
from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.game_logic.rules.core.post_merge.post_merge_rule import PostMergeRule
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.core.scoring.track_cleared_lines_rule import ClearedLinesTracker
from tetris.game_logic.rules.core.scoring.track_score_rule import ScoreTracker
from tetris.game_logic.rules.core.spawn.spawn import SpawnRule
from tetris.game_logic.rules.core.spawn.synchronized_spawn import SynchronizedSpawning
from tetris.game_logic.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.game_logic.rules.runtime.sound_toggle import SoundToggleRule
from tetris.game_logic.rules.special.powerup import PowerupRule
from tetris.game_logic.runtime import Runtime
from tetris.game_logic.sound_manager import SoundManager
from tetris.ui.cli import CLI

if TYPE_CHECKING:
    from tetris.game_logic.interfaces.audio_output import AudioOutput
    from tetris.game_logic.interfaces.rule import Rule
    from tetris.game_logic.interfaces.runtime_rule import RuntimeRule

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

type AudioBackendParameter = Literal["pygame", "playsound3", "winsound"]


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
        "used for each game. The special value 'same' can be provided to use the same random seed for all games."
    ),
)
@click.option(
    "--sounds",
    default="tetris_nes",
    show_default=True,
    help=(
        "Name the sound pack to use for sounds (directory with sound files in data/sounds directory, "
        "or a known online sound pack (currently just 'tetris_nes')). Use the special value 'off' to disable sounds."
    ),
)
@click.option(
    "--sounded-game",
    type=click.IntRange(min=0),
    multiple=True,
    help=(
        "Index of a game for which to produce sounds (0-based, i.e. first game is index 0). "
        "Can be specified multiple times to enable sound for multiple games. "
        "If not specified, all games have sound. "
        "Note: It is recommended to not enable sound for more than 4 games at once (especially with fast-playing bots)."
    ),
)
@click.option(
    "--audio-backend",
    type=click.Choice(["pygame", "playsound3", "winsound"], case_sensitive=False),
    default="pygame",
    show_default=True,
    help=(
        "Backend to use for playing sounds. Pygame is recommended. "
        "Note: 'winsound' only works on Windows, and only with 'wav' files."
    ),
)
@click.option(
    "--track-performance/--no-track-performance",
    default=False,
    show_default=True,
    help="Enable/disable tracking of performance metrics.",
)
@click.option(
    "--ghost-block/--no-ghost-block", default=True, show_default=True, help="Enable/disable display of ghost block."
)
@click.option("--powerups/--no-powerups", default=True, show_default=True, help="Enable/disable power-ups.")
@click.option(
    "--fuzz-test",
    is_flag=True,
    default=False,
    help=(
        "Flag to run a large-scale automatic fuzz testing session. Will overwrite some other options. "
        "This runs 8 games controlled by bots, with tetris99 and powerups enabled, at max FPS, "
        "with sounds only from game 0."
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
    sounds: str,
    sounded_game: tuple[int, ...],
    audio_backend: AudioBackendParameter,
    track_performance: bool,
    ghost_block: bool,
    powerups: bool,
    fuzz_test: bool,
) -> None:
    """Play Tetris with configurable rules and controllers."""
    runtime_controller: Controller | None = None
    tetris99_self_targeting_when_alone = False
    if fuzz_test:
        num_games = 8
        controller = ("bot",)
        tetris99 = True
        synchronize_spawning = False
        process_pool = True
        fps = 1_000_000  # it will just run as fast as it can
        seed = ()
        sounded_game = (0,)
        track_performance = False
        ghost_block = True
        powerups = True
        # make sure a new game is automatically started after all are game over
        runtime_controller = StubController(Action(confirm=True), mode="press_repeatedly")
        tetris99_self_targeting_when_alone = True

    boards, controllers = _create_boards_and_controllers(
        board_size=board_size, num_games_parameter=num_games, controllers_parameter=controller, powerups=powerups
    )

    if not all(0 <= game_index < len(boards) for game_index in sounded_game):
        invalid_indices = [str(game_index) for game_index in sounded_game if 0 <= game_index < len(boards)]
        msg = (
            f"Invalid game indices provided to --sounded-game: {', '.join(invalid_indices)}. "
            f"Indices must be within [0, {len(boards) - 1}] as there are {len(boards)} games."
        )
        raise click.BadParameter(msg)

    any_bots = any(isinstance(controller, HeuristicBotController | BotAssistedController) for controller in controllers)

    seeds = _create_seeds(seed_parameters=seed, num_games=len(boards))
    block_selection_fns = [getattr(SpawnRule, f"{block_selection}_selection_fn")(seed) for seed in seeds]

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
            tetris99_self_targeting_when_alone=tetris99_self_targeting_when_alone,
            synchronize_spawning=synchronize_spawning,
            ghost_block=ghost_block,
            powerups=powerups,
        )

        sound_manager = (
            None
            if sounds == "off"
            else _create_sound_manager(
                sounds=sounds, sounded_game_indices=sounded_game or None, audio_backend=audio_backend
            )
        )

        runtime = _create_runtime(
            games,
            fps=fps,
            sound_manager=sound_manager,
            track_performance=track_performance,
            controller=runtime_controller,
        )

        DEPENDENCY_MANAGER.wire_up(runtime=runtime, games=games)

        runtime.run()


def _create_boards_and_controllers(
    board_size: tuple[int, int],
    num_games_parameter: int | None,
    controllers_parameter: tuple[ControllerParameter, ...],
    *,
    powerups: bool,
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
                _create_controller(controller_parameter, board, powerups=powerups)
                for controller_parameter, board in zip(controllers_parameter, boards, strict=True)
            ]
        else:
            # If controller is not specified, a single game with keyboard controller is created.
            from tetris.controllers.keyboard.pynput import PynputKeyboardController

            controller: Controller = PynputKeyboardController()
            if powerups:
                controller = BotAssistedController(
                    controller, HeuristicBotController(boards[0]), allow_manual_activation=False
                )

            controllers = [controller]
    else:  # noqa: PLR5501
        # Case 2: --num-games is specified
        if controllers_parameter:
            # If controller is specified once, all games will use that same controller.
            if len(controllers_parameter) == 1:
                controllers_parameter *= num_games

            # If specified multiple times, specifies the first N games' controllers, with the rest being filled up with
            # bots.
            controllers = [
                _create_controller(controller_parameter, board, powerups=powerups)
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


def _create_controller(controller_parameter: ControllerParameter, board: Board, *, powerups: bool) -> Controller:
    if controller_parameter == "bot":
        return HeuristicBotController(board)

    manually_bot_assisted = controller_parameter.endswith("+")

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

    if manually_bot_assisted:
        controller = BotAssistedController(controller, HeuristicBotController(board), allow_manual_activation=True)
    elif powerups:
        # one powerup is bot-assistance: we need to make the controller a bot-assisted one, without manual activateion
        controller = BotAssistedController(controller, HeuristicBotController(board), allow_manual_activation=False)

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


def _create_games(  # noqa: PLR0913
    boards: list[Board],
    controllers: list[Controller],
    block_selection_fns: list[Callable[[], Block]],
    *,
    tetris99: bool,
    tetris99_self_targeting_when_alone: bool = False,
    synchronize_spawning: bool,
    ghost_block: bool,
    powerups: bool,
) -> list[Game]:
    games: list[Game] = []

    for idx, (controller, board, block_selection_fn) in enumerate(
        zip(controllers, boards, block_selection_fns, strict=True)
    ):
        DEPENDENCY_MANAGER.current_game_index = idx

        # not useless; Dependency manager will keep track of them and subscribe them to line clear events and each
        # other, and publish lines and score will be published to the ui aggregator
        ClearedLinesTracker()
        LevelTracker()
        ScoreTracker()

        if isinstance(controller, Callback | Subscriber | Publisher):
            controller.game_index = idx

        if isinstance(controller, BotAssistedController):
            controller.bot_controller.game_index = idx

        rule_sequence = _create_rules_and_callbacks(
            num_games=len(boards),
            create_tetris_99_rule=tetris99,
            tetris99_self_targeting_when_alone=tetris99_self_targeting_when_alone,
            synchronize_spawning=synchronize_spawning,
            block_selection_fn=block_selection_fn,
            powerups=powerups,
        )

        games.append(
            Game(board=board, controller=controller, rule_sequence=rule_sequence, show_ghost_block=ghost_block)
        )

    return games


def _create_rules_and_callbacks(  # noqa: PLR0913
    num_games: int,
    *,
    create_tetris_99_rule: bool,
    tetris99_self_targeting_when_alone: bool = False,
    synchronize_spawning: bool,
    block_selection_fn: Callable[[], Block] = Block.create_random,
    powerups: bool,
) -> RuleSequence:
    """Get rules relevant for one instance of a game.

    Optionally also create (but don't return) callbacks to be added to game's/runtime's callback collection by wire-up
    function.

    Args:
        num_games: Total number of games being created overall.
        create_tetris_99_rule: Whether to create a Tetris99Rule in case there are multiple games.
        tetris99_self_targeting_when_alone: Whether to enable self-targeting when no other games are alive.
        synchronize_spawning: Whether to synchronize spawning in case there are multiple games.
        block_selection_fn: Function to select blocks to spawn.
        powerups: Whether to enable power-ups.

    Returns:
        A RuleSequence to be passed to the Game.
    """
    synchronize_spawning = synchronize_spawning and num_games > 1

    rules: list[Rule] = [MoveRule(), RotateRule(), DropMergeRule(), PostMergeRule(), SpawnRule(block_selection_fn)]

    if powerups:
        rules.append(PowerupRule())

    if create_tetris_99_rule and num_games > 1:
        rules.append(
            Tetris99Rule(
                target_idxs=list(set(range(num_games)) - {DEPENDENCY_MANAGER.current_game_index}),
                self_targeting_when_alone=tetris99_self_targeting_when_alone,
            )
        )

    if synchronize_spawning:
        SynchronizedSpawning()  # not useless; will be added to DEPENDENCY_MANAGER.all_callbacks

    return RuleSequence(rules)


def _create_sound_manager(
    sounds: str, sounded_game_indices: Collection[int] | None, audio_backend: AudioBackendParameter
) -> SoundManager | None:
    match audio_backend:
        case "playsound3":
            from tetris.audio_outputs.playsound3 import Playsound3AudioOutput

            audio_output: AudioOutput = Playsound3AudioOutput()
        case "pygame":
            from tetris.audio_outputs.pygame import PygameAudioOutput

            audio_output = PygameAudioOutput()
        case "winsound":
            from tetris.audio_outputs.winsound import WinsoundAudioOutput

            audio_output = WinsoundAudioOutput()

    return SoundManager(audio_output, sound_pack=sounds, game_indices=sounded_game_indices)


def _create_runtime(
    games: list[Game],
    *,
    controller: Controller | None = None,
    fps: float = 60,
    sound_manager: SoundManager | None = None,
    track_performance: bool = False,
) -> Runtime:
    rules: list[RuntimeRule] = []

    DEPENDENCY_MANAGER.current_game_index = DependencyManager.RUNTIME_INDEX
    if sound_manager is not None:
        # NOTE: SoundManager's game_index is never actually used, but for consistency's sake still change it
        sound_manager.game_index = DependencyManager.RUNTIME_INDEX

        rules.append(SoundToggleRule(sound_manager))

    if track_performance:
        TrackPerformanceCallback(fps)  # not useless; will be added to DEPENDENCY_MANAGER.all_callbacks

    ui = CLI()
    clock = SimpleClock(fps)

    if controller is None:
        try:
            from tetris.controllers.keyboard.pynput import PynputKeyboardController
        except Exception:  # noqa: BLE001
            controller = StubController(Action())
        else:
            controller = PynputKeyboardController()

    return Runtime(
        ui=ui,
        clock=clock,
        games=games,
        controller=controller,
        sound_manager=sound_manager,
        rule_sequence=RuntimeRuleSequence(rules),
    )
