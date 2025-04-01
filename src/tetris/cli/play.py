# commands/play.py
import logging

import click

# --- Assuming your project structure allows these imports ---
from tetris.cli.helpers import (
    create_rules_and_callbacks_for_game,
    get_available_controller_specs,
    manage_process_pool,
    parse_and_create_controllers,
)
from tetris.clock.simple import SimpleClock
from tetris.controllers.keyboard.pynput import PynputKeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER, DependencyManager
from tetris.game_logic.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.game_logic.runtime import Runtime
from tetris.ui.cli import CLI

# --- Constants ---
LOGGER = logging.getLogger(__name__)
DEFAULT_FPS = 60.0
FUZZ_TEST_FPS = 100000.0  # Effectively unlimited
DEFAULT_SPAWN_DELAY = 0
DEFAULT_BOARD_WIDTH = 10
DEFAULT_BOARD_HEIGHT = 20


@click.command()
@click.option("-n", "--num-games", type=int, default=1, show_default=True, help="Number of concurrent games.")
@click.option(
    "-c",
    "--controllers",
    multiple=True,
    help=f"Controller types (space-separated). Choices: {', '.join(get_available_controller_specs())}. Defaults based on num_games.",
)
@click.option(
    "--use-tetris99/--no-tetris99", default=True, show_default=True, help="Enable/disable Tetris99 attack rules."
)
@click.option(
    "--bot-process-pool/--no-bot-process-pool",
    default=True,
    show_default=True,
    help="Use a process pool for Bot controllers.",
)
@click.option(
    "--bot-ensure-consistent/--no-bot-ensure-consistent",
    default=False,
    show_default=True,
    help="Ensure consistent bot behavior with process pool (can be slower).",
)
@click.option("--fps", type=float, default=DEFAULT_FPS, show_default=True, help="Target frames per second.")
@click.option(
    "--spawn-delay",
    type=int,
    default=DEFAULT_SPAWN_DELAY,
    show_default=True,
    help="Frames after merge before next spawn.",
)
@click.option("--seed", type=int, default=None, help="Master seed for block generation (for reproducibility).")
@click.option(
    "--board-width", type=int, default=DEFAULT_BOARD_WIDTH, show_default=True, help="Width of the game board."
)
@click.option(
    "--board-height", type=int, default=DEFAULT_BOARD_HEIGHT, show_default=True, help="Height of the game board."
)
@click.option(
    "--fuzz-test",
    is_flag=True,
    default=False,
    help="Run in high-speed fuzz testing mode (bots, high FPS, process pool).",
)
def play(
    num_games: int,
    controllers: tuple[str, ...],
    use_tetris99: bool,
    bot_process_pool: bool,
    bot_ensure_consistent: bool,
    fps: float,
    spawn_delay: int,
    seed: int | None,
    board_width: int,
    board_height: int,
    fuzz_test: bool,
) -> None:
    """Play Tetris with configurable rules and controllers."""
    active_controllers = controllers
    active_fps = fps
    active_bot_process_pool = bot_process_pool

    if fuzz_test:
        LOGGER.warning("Running in FUZZ TEST mode!")
        active_controllers = ("bot",) * num_games
        active_fps = FUZZ_TEST_FPS
        active_bot_process_pool = True  # Force process pool for fuzz testing
        # Keep other settings like tetris99 as per user flag or default

    # --- Setup ---
    DEPENDENCY_MANAGER.reset()
    LOGGER.info("Starting game setup for %d game(s).", num_games)
    boards = [Board.create_empty(board_height, board_width) for _ in range(num_games)]

    # Manage process pool context
    pool_manager = manage_process_pool(active_bot_process_pool)
    try:
        with pool_manager as process_pool_instance:  # Enter the context
            # Create controllers
            controller_instances = parse_and_create_controllers(
                active_controllers,
                num_games,
                boards,
                use_process_pool=active_bot_process_pool,
                ensure_consistent=bot_ensure_consistent,
                process_pool_instance=process_pool_instance,
            )

            # Create games
            games = _create_games(
                num_games,
                boards,
                controller_instances,
                spawn_delay,
                seed,
                use_tetris99,
            )

            # Create Runtime
            runtime = _create_runtime(games, active_fps)

            # --- Wire Dependencies ---
            LOGGER.info("Wiring dependencies...")
            DEPENDENCY_MANAGER.wire_up(runtime=runtime, games=games)

            # --- Run ---
            LOGGER.info("Starting runtime with %d game(s) at %.1f FPS. Press Ctrl+C to exit.", num_games, active_fps)
            runtime.run()

    except Exception:
        LOGGER.exception("An error occurred during gameplay setup or execution.")
        # traceback.print_exc() # Logger should capture this
    finally:
        # Process pool is automatically shut down by the 'with' statement
        DEPENDENCY_MANAGER.reset()
        LOGGER.info("Play command finished.")


def _create_games(
    num_games: int,
    boards: list[Board],
    controllers: list[Controller],
    spawn_delay: int,
    seed: int | None,
    use_tetris_99: bool,
) -> list[Game]:
    """Helper to create Game instances."""
    games: list[Game] = []
    for i in range(num_games):
        rule_sequence = create_rules_and_callbacks_for_game(
            game_index=i, num_games=num_games, spawn_delay=spawn_delay, seed=seed, use_tetris_99_rules=use_tetris_99
        )
        games.append(Game(board=boards[i], controller=controllers[i], rule_sequence=rule_sequence))
        LOGGER.debug("Created game %d.", i)
    return games


def _create_runtime(games: list[Game], fps: float) -> Runtime:
    """Helper to create the Runtime instance."""
    DEPENDENCY_MANAGER.current_game_index = DependencyManager.RUNTIME_INDEX
    # Register runtime-specific callbacks
    TrackPerformanceCallback(fps)
    runtime_controller = PynputKeyboardController()  # For pause/menu interaction
    LOGGER.debug("Creating runtime with UI: %s, Clock: %s", CLI.__name__, SimpleClock.__name__)
    return Runtime(ui=CLI(), clock=SimpleClock(fps), games=games, controller=runtime_controller)
