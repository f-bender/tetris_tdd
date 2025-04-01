# main.py
import ast
import contextlib
import logging
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import click
import numpy as np
from numpy.typing import NDArray

# --- Assuming your project structure allows these imports ---
# (Adjust paths if necessary, e.g., if 'tetris' is a top-level package)
try:
    from tetris.clock.simple import SimpleClock
    from tetris.controllers.heuristic_bot.controller import HeuristicBotController
    from tetris.controllers.heuristic_bot.heuristic import Heuristic
    from tetris.controllers.keyboard.pynput import PynputKeyboardController
    from tetris.game_logic.components import Board
    from tetris.game_logic.game import Game
    from tetris.game_logic.interfaces.callback import Callback
    from tetris.game_logic.interfaces.controller import Controller
    from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER, DependencyManager
    from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
    from tetris.game_logic.interfaces.rule_sequence import RuleSequence
    from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
    from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
    from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
    from tetris.game_logic.rules.monitoring.track_performance_rule import TrackPerformanceCallback
    from tetris.game_logic.rules.monitoring.track_score_rule import TrackScoreRule
    from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule
    from tetris.game_logic.rules.special.parry_rule import ParryRule
    from tetris.game_logic.runtime import Runtime
    from tetris.genetic_algorithm import GeneticAlgorithm
    from tetris.heuristic_gym.detailed_heuristic_evaluator import DetailedHeuristicEvaluator
    from tetris.heuristic_gym.evaluators.evaluator import EvaluatorImpl
    from tetris.heuristic_gym.heuristic_gym import HeuristicGym
    from tetris.logging_config import configure_logging
    from tetris.space_filling_coloring import drawer
    from tetris.space_filling_coloring.concurrent_fill_and_colorize import fill_and_colorize
    from tetris.space_filling_coloring.four_colorizer import FourColorizer, UnableToColorizeError
    from tetris.space_filling_coloring.fuzz_test_concurrent_fill_and_colorize import fuzz_test as fill_fuzz_test
    from tetris.space_filling_coloring.tetromino_space_filler import NotFillableError, TetrominoSpaceFiller
    from tetris.ui.cli import CLI

    # Import GamepadController if available
    try:
        from inputs import devices

        from tetris.controllers.gamepad import GamepadController

        GAMEPAD_AVAILABLE = True
    except ImportError:
        GamepadController = None  # type: ignore
        GAMEPAD_AVAILABLE = False

    # Import other necessary components if needed
    from tetris.controllers.heuristic_bot.heuristic import mutated_heuristic  # For GA default

except ImportError as e:
    print(f"Error importing Tetris components: {e}")
    print("Please ensure the script is run from the correct directory or the 'tetris' package is installed.")
    sys.exit(1)

# --- Constants ---
LOGGER = logging.getLogger(__name__)
DEFAULT_FPS = 60.0
FUZZ_TEST_FPS = 100000.0  # Effectively unlimited for fuzzing play
DEFAULT_SPAWN_DELAY = 0  # Match default in example main
DEFAULT_BOARD_WIDTH = 10
DEFAULT_BOARD_HEIGHT = 20
DEFAULT_EVALUATOR_BOARD_WIDTH = 10
DEFAULT_EVALUATOR_BOARD_HEIGHT = 15
DEFAULT_EVALUATOR_MAX_FRAMES = 300_000  # ~ 1.4 hours at 60fps

# --- Helper Functions ---


def _literal_eval_heuristic(heuristic_repr: str) -> Heuristic:
    """Safely evaluate a Heuristic string representation."""
    try:
        # Basic check for Heuristic(...) structure
        if not (heuristic_repr.startswith("Heuristic(") and heuristic_repr.endswith(")")):
            msg = "Invalid Heuristic string format."
            raise ValueError(msg)

        # Attempt to parse the arguments within the parentheses
        # Replace Heuristic( with {' and ) with } and = with ': and , with ', '
        # This transforms "Heuristic(a=1, b=0.5)" into "{'a':1, 'b':0.5}"
        dict_literal = heuristic_repr[len("Heuristic(") : -1]
        # Handle case with no arguments
        if not dict_literal.strip():
            args_dict = {}
        else:
            # More robust parsing than simple replace, handles spaces etc.
            # We'll use ast.parse to build the dict safely
            # Example: "a = 1 , b = - 0.5"
            # Needs careful construction to be valid python dict literal inside {}
            node = ast.parse(f"dict({dict_literal})", mode="eval")
            args_dict = ast.literal_eval(node)  # Safely evaluate the expression

        # Validate keys are actual Heuristic parameters (optional but good practice)
        valid_keys = Heuristic._fields
        for key in args_dict:
            if key not in valid_keys:
                msg = f"Invalid Heuristic parameter: {key}"
                raise ValueError(msg)

        return Heuristic(**args_dict)
    except (ValueError, SyntaxError, TypeError) as e:
        msg = f"Invalid Heuristic string '{heuristic_repr}': {e}"
        raise click.BadParameter(msg)


def _parse_controllers(
    controller_specs: tuple[str, ...],
    num_games: int,
    boards: list[Board],
    bot_lightning: bool,
    bot_process_pool_ctx: ProcessPoolExecutor | contextlib.nullcontext,
    bot_heuristic: Heuristic,
    bot_ensure_consistent: bool,
) -> list[Controller]:
    """Parses controller specifications and creates Controller instances."""
    controllers: list[Controller] = []
    specs = list(controller_specs)  # Make mutable

    available_gamepads = 0
    if GAMEPAD_AVAILABLE:
        available_gamepads = len(devices.gamepads)

    gamepad_idx_counter = 0

    # Default controller if none specified
    if not specs:
        specs.append("keyboard")
        # If more than one game, add bots for the rest by default
        if num_games > 1:
            specs.extend(["bot"] * (num_games - 1))

    if len(specs) != num_games:
        # If only one spec given for multiple games, replicate it
        if len(specs) == 1:
            spec_to_replicate = specs[0]
            # Special case: if 'gamepad' specified once for multiple games, try assigning unique gamepads
            if spec_to_replicate == "gamepad":
                if available_gamepads >= num_games:
                    specs = ["gamepad"] * num_games
                else:
                    # Not enough gamepads, fill rest with bots
                    specs = ["gamepad"] * available_gamepads + ["bot"] * (num_games - available_gamepads)
            else:
                specs = [spec_to_replicate] * num_games
        else:
            msg = (
                f"Number of controllers specified ({len(specs)}) must match number of games ({num_games}). "
                f"Provide {num_games} controller types separated by commas, or provide one type to be used for all."
            )
            raise click.BadParameter(msg)

    process_pool_instance = (
        bot_process_pool_ctx.__enter__() if isinstance(bot_process_pool_ctx, ProcessPoolExecutor) else None
    )

    for i, spec in enumerate(specs):
        board = boards[i]
        spec_lower = spec.lower()
        if spec_lower == "keyboard":
            controllers.append(PynputKeyboardController.arrow_keys())
        elif spec_lower == "wasd":
            controllers.append(PynputKeyboardController.wasd())
        elif spec_lower == "vim":
            controllers.append(PynputKeyboardController.vim())
        elif spec_lower == "bot":
            controllers.append(
                HeuristicBotController(
                    board,
                    heuristic=bot_heuristic,
                    lightning_mode=bot_lightning,
                    process_pool=process_pool_instance,
                    ensure_consistent_behaviour=bot_ensure_consistent,
                )
            )
        elif spec_lower == "gamepad":
            if not GAMEPAD_AVAILABLE:
                msg = "Gamepad support not available (failed to import 'inputs' library or no gamepads found)."
                raise click.ClickException(msg)
            if gamepad_idx_counter >= available_gamepads:
                msg = f"Requested gamepad controller {gamepad_idx_counter + 1}, but only {available_gamepads} found."
                raise click.ClickException(msg)
            controllers.append(GamepadController(gamepad_index=gamepad_idx_counter))
            gamepad_idx_counter += 1
        else:
            msg = f"Unknown controller type: '{spec}'. Choices: keyboard, wasd, vim, bot, gamepad."
            raise click.BadParameter(msg)

    return controllers


def _parse_holes(holes_str: str | None) -> list[tuple[int, int, int, int]]:
    """Parses a string like "y1,x1,y2,x2;y1,x1,y2,x2" into a list of tuples."""
    if not holes_str:
        return []
    holes = []
    try:
        hole_definitions = holes_str.split(";")
        for hole_def in hole_definitions:
            parts = [int(p.strip()) for p in hole_def.split(",")]
            if len(parts) != 4:
                msg = "Each hole definition must have 4 comma-separated integers."
                raise ValueError(msg)
            y1, x1, y2, x2 = parts
            if not (0 <= y1 < y2 and 0 <= x1 < x2):
                msg = f"Invalid hole coordinates (must be y1 < y2 and x1 < x2): {parts}"
                raise ValueError(msg)
            holes.append((y1, x1, y2, x2))
        return holes
    except ValueError as e:
        msg = f"Invalid format for --holes: '{holes_str}'. Error: {e}. Expected format: 'y1,x1,y2,x2;y1,x1,y2,x2;...'."
        raise click.BadParameter(msg)


def _generate_space(
    width: int, height: int, holes: list[tuple[int, int, int, int]], inverted: bool
) -> NDArray[np.bool_]:
    """Generates the initial boolean space array."""
    if not (width > 0 and height > 0):
        msg = "Width and height must be positive."
        raise click.BadParameter(msg)

    space = np.ones((height, width), dtype=bool)

    for y1, x1, y2, x2 in holes:
        if not (y2 <= height and x2 <= width):
            msg = f"Hole coordinates {y1},{x1},{y2},{x2} are outside the space dimensions {height}x{width}."
            raise click.BadParameter(msg)
        space[y1:y2, x1:x2] = False

    if inverted:
        space = ~space

    return space


def _create_rules_and_callbacks_for_game(
    game_index: int, num_games: int, spawn_delay: int, seed: int | None, use_tetris_99_rules: bool
) -> RuleSequence:
    """Creates rules and registers callbacks for a specific game instance."""
    DEPENDENCY_MANAGER.current_game_index = game_index

    # Register callbacks (they add themselves to DEPENDENCY_MANAGER)
    TrackScoreRule(header=f"Game {game_index}")

    # Create Spawn Strategy with seed if provided
    if seed is not None:
        game_seed = seed + game_index  # Simple way to get different seeds per game from one master seed
        spawn_strategy = SpawnStrategyImpl.from_shuffled_bag(seed=game_seed)
    else:
        spawn_strategy = SpawnStrategyImpl()  # Use default random

    rules: list[Any] = [
        MoveRule(),
        RotateRule(),
        SpawnDropMergeRule(spawn_delay=spawn_delay, spawn_strategy=spawn_strategy),
        ParryRule(),
    ]

    if use_tetris_99_rules and num_games > 1:
        target_idxs = list(set(range(num_games)) - {game_index})
        rules.append(Tetris99Rule(target_idxs=target_idxs))

    return RuleSequence(rules)


# --- Click CLI Definition ---


@click.group()
def main() -> None:
    """A command-line interface for the Tetris project."""
    configure_logging()  # Configure logging once when the CLI starts


@main.command()
@click.option("-n", "--num-games", type=int, default=1, show_default=True, help="Number of concurrent games to play.")
@click.option(
    "-c",
    "--controllers",
    multiple=True,
    help="Controller types for each game (comma-separated or repeat option). Choices: keyboard, wasd, vim, bot, gamepad. Defaults based on num_games.",
)
@click.option(
    "--use-tetris99/--no-tetris99", default=True, show_default=True, help="Enable/disable Tetris99 attack rules."
)
@click.option(
    "--bot-heuristic",
    type=str,
    default="Heuristic()",
    show_default=True,
    help='Heuristic parameters for Bot controllers, e.g., "Heuristic(a=1,b=-0.5)".',
)
@click.option(
    "--bot-lightning/--no-bot-lightning",
    default=False,
    show_default=True,
    help="Enable lightning mode for Bot controllers (faster, less realistic).",
)
@click.option(
    "--bot-process-pool/--no-bot-process-pool",
    default=True,
    show_default=True,
    help="Use a process pool for Bot controllers (can improve performance).",
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
    help="Frames to wait after merge before spawning next block.",
)
@click.option(
    "--seed", type=int, default=None, help="Master seed for block generation across all games (for reproducibility)."
)
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
    help="Run in high-speed fuzz testing mode (uses bots, lightning, high FPS).",
)
def play(
    num_games: int,
    controllers: tuple[str, ...],
    use_tetris99: bool,
    bot_heuristic: str,
    bot_lightning: bool,
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
    if fuzz_test:
        click.echo("Starting Fuzz Test Mode...")
        controllers = ("bot",) * num_games
        bot_lightning = True
        # bot_process_pool can remain as chosen, might be faster with it
        fps = FUZZ_TEST_FPS
        # Maybe disable tetris99 for pure speed focus? Optional.
        # use_tetris99 = False

    # --- Setup ---
    DEPENDENCY_MANAGER.reset()  # Ensure clean state

    boards = [Board.create_empty(board_height, board_width) for _ in range(num_games)]

    parsed_heuristic = _literal_eval_heuristic(bot_heuristic)

    pool_context: ProcessPoolExecutor | contextlib.nullcontext = (
        ProcessPoolExecutor() if bot_process_pool else contextlib.nullcontext()
    )

    try:
        # Create controllers within the pool context if needed
        controller_instances = _parse_controllers(
            controllers,
            num_games,
            boards,
            bot_lightning=bot_lightning,
            bot_process_pool_ctx=pool_context,  # Pass the context manager
            bot_heuristic=parsed_heuristic,
            bot_ensure_consistent=bot_ensure_consistent,
        )

        # Create games, registering rules and callbacks
        games: list[Game] = []
        for i in range(num_games):
            rule_sequence = _create_rules_and_callbacks_for_game(
                game_index=i, num_games=num_games, spawn_delay=spawn_delay, seed=seed, use_tetris_99_rules=use_tetris99
            )
            # Register the controller itself if it's a subscriber/callback etc.
            # Note: Dependency Manager handles wiring based on registration during init
            # but we need to ensure game_index is set if it's a Callback/Subscriber/Publisher
            controller = controller_instances[i]
            if isinstance(controller, Callback | Subscriber | Publisher):
                # Should have been set during controller parsing if it's a bot
                # Re-setting here just in case, though ideally parsing handles it
                if hasattr(controller, "game_index"):
                    # HeuristicBotController handles its own game_index via board reference usually
                    pass  # Assume it's handled correctly
                else:
                    # This case shouldn't happen with current controllers
                    LOGGER.warning(f"Controller {type(controller)} might need game_index but it's not set.")

            games.append(
                Game(
                    board=boards[i],
                    controller=controller_instances[i],
                    rule_sequence=rule_sequence,
                    # callback_collection will be set by wire_up
                )
            )

        # Create Runtime
        DEPENDENCY_MANAGER.current_game_index = DependencyManager.RUNTIME_INDEX
        # Register runtime-specific callbacks
        TrackPerformanceCallback(fps)
        runtime_controller = PynputKeyboardController()  # For pause/menu interaction

        runtime = Runtime(
            ui=CLI(),
            clock=SimpleClock(fps),
            games=games,
            controller=runtime_controller,
            # callback_collection will be set by wire_up
        )

        # --- Wire Dependencies ---
        DEPENDENCY_MANAGER.wire_up(runtime=runtime, games=games)

        # --- Run ---
        click.echo(f"Starting {num_games} game(s)... Press Ctrl+C to exit.")
        if bot_process_pool and isinstance(pool_context, ProcessPoolExecutor):
            click.echo("Using Process Pool for Bots.")
        runtime.run()

    except Exception:
        LOGGER.exception("An error occurred during gameplay.")
        traceback.print_exc()
    finally:
        if isinstance(pool_context, ProcessPoolExecutor):
            # Ensure pool is shut down if context wasn't used with 'with'
            if hasattr(pool_context, "_shutdown"):  # Check if pool still exists
                click.echo("Shutting down process pool...")
                pool_context.shutdown(wait=True)
        DEPENDENCY_MANAGER.reset()  # Clean up dependency manager state
        click.echo("Game finished.")


@main.command()
@click.option(
    "--population-size", type=int, default=100, show_default=True, help="Number of heuristics in each generation."
)
@click.option("--generations", type=int, default=None, help="Number of generations to run (default: run indefinitely).")
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    default="./.checkpoints",
    show_default=True,
    help="Directory to save and load checkpoints.",
)
@click.option(
    "--continue-from",
    type=str,
    default=None,
    help="Path to a specific checkpoint file, or 'latest' to use the most recent in checkpoint-dir.",
)
@click.option(
    "--evaluator-board-width",
    type=int,
    default=DEFAULT_EVALUATOR_BOARD_WIDTH,
    show_default=True,
    help="Board width for evaluation games.",
)
@click.option(
    "--evaluator-board-height",
    type=int,
    default=DEFAULT_EVALUATOR_BOARD_HEIGHT,
    show_default=True,
    help="Board height for evaluation games.",
)
@click.option(
    "--evaluator-max-frames",
    type=int,
    default=DEFAULT_EVALUATOR_MAX_FRAMES,
    show_default=True,
    help="Maximum frames per evaluation game.",
)
@click.option(
    "--evaluator-process-pool/--no-evaluator-process-pool",
    default=True,
    show_default=True,
    help="Use a process pool for the evaluator.",
)
@click.option(
    "--ga-survival-rate",
    type=float,
    default=0.5,
    show_default=True,
    help="Fraction of population surviving each generation.",
)
@click.option(
    "--ga-elitism-factor",
    type=float,
    default=1.0,
    show_default=True,
    help="Exponent favoring fitter individuals for survival.",
)
@click.option(
    "--initial-heuristic",
    type=str,
    default="Heuristic()",
    show_default=True,
    help="Heuristic for the initial population if not continuing.",
)
def train(
    population_size: int,
    generations: int | None,
    checkpoint_dir: str,
    continue_from: str | None,
    evaluator_board_width: int,
    evaluator_board_height: int,
    evaluator_max_frames: int,
    evaluator_process_pool: bool,
    ga_survival_rate: float,
    ga_elitism_factor: float,
    initial_heuristic: str,
) -> None:
    """Train HeuristicBots using a genetic algorithm."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    pool_context: ProcessPoolExecutor | contextlib.nullcontext = (
        ProcessPoolExecutor() if evaluator_process_pool else contextlib.nullcontext()
    )

    try:
        process_pool_instance = pool_context.__enter__() if isinstance(pool_context, ProcessPoolExecutor) else None

        evaluator = EvaluatorImpl(
            board_size=(evaluator_board_height, evaluator_board_width),
            max_evaluation_frames=evaluator_max_frames,
            process_pool=process_pool_instance,
        )

        # Using functools.partial is recommended over lambda for pickling
        ga = GeneticAlgorithm(
            mutator=partial(mutated_heuristic),  # Use default mutator
            survival_rate=ga_survival_rate,
            elitism_factor=ga_elitism_factor,
        )

        gym = HeuristicGym(
            population_size=population_size, evaluator=evaluator, genetic_algorithm=ga, checkpoint_dir=checkpoint_path
        )

        if continue_from:
            if continue_from.lower() == "latest":
                click.echo(f"Attempting to continue from latest checkpoint in '{checkpoint_path}'...")
                try:
                    # HeuristicGym handles finding the latest .pkl file
                    # Pass overwrite options if needed, but we'll rely on defaults or loaded config here
                    HeuristicGym.continue_from_latest_checkpoint(
                        checkpoint_dir=checkpoint_path,
                        new_population_size=population_size,  # Allow overriding loaded pop size
                        num_generations=generations,
                        # Add **kwargs_to_overwrite if you want CLI args to override checkpoint values
                    )
                except ValueError as e:
                    msg = f"Could not continue from latest: {e}"
                    raise click.ClickException(msg)
            else:
                checkpoint_file_path = Path(continue_from)
                if not checkpoint_file_path.is_file():
                    msg = f"Checkpoint file not found: {checkpoint_file_path}"
                    raise click.BadParameter(msg)
                click.echo(f"Attempting to continue from checkpoint '{checkpoint_file_path}'...")
                try:
                    with checkpoint_file_path.open("rb") as f:
                        HeuristicGym.continue_from_checkpoint(
                            checkpoint_file=f,
                            checkpoint_dir=checkpoint_path,
                            new_population_size=population_size,
                            num_generations=generations,
                            # Add **kwargs_to_overwrite if needed
                        )
                except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as e:
                    msg = f"Failed to load checkpoint '{checkpoint_file_path}': {e}"
                    raise click.ClickException(msg)
        else:
            click.echo("Starting new training run...")
            initial_heuristic_instance = _literal_eval_heuristic(initial_heuristic)
            gym.run(
                initial_population=[initial_heuristic_instance],  # Start with one, GA will expand
                num_generations=generations,
            )

    except Exception:
        LOGGER.exception("An error occurred during training.")
        traceback.print_exc()
    finally:
        if isinstance(pool_context, ProcessPoolExecutor) and hasattr(pool_context, "_shutdown"):
            click.echo("Shutting down process pool...")
            pool_context.shutdown(wait=True)
        click.echo("Training finished.")


@main.command()
@click.argument("heuristic", type=str)
@click.option("--num-games", type=int, default=50, show_default=True, help="Number of games to play for evaluation.")
@click.option("--seed", type=int, default=42, show_default=True, help="Seed for the evaluation run randomness.")
@click.option(
    "--report-file",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    default="./reports/report.csv",
    show_default=True,
    help="CSV file to append evaluation results.",
)
@click.option(
    "--evaluator-board-width",
    type=int,
    default=DEFAULT_EVALUATOR_BOARD_WIDTH,
    show_default=True,
    help="Board width for evaluation games.",
)
@click.option(
    "--evaluator-board-height",
    type=int,
    default=DEFAULT_EVALUATOR_BOARD_HEIGHT,
    show_default=True,
    help="Board height for evaluation games.",
)
@click.option(
    "--evaluator-max-frames",
    type=int,
    default=DEFAULT_EVALUATOR_MAX_FRAMES,
    show_default=True,
    help="Maximum frames per evaluation game.",
)
@click.option(
    "--evaluator-process-pool/--no-evaluator-process-pool",
    default=True,
    show_default=True,
    help="Use a process pool for the evaluator.",
)
@click.option(
    "--extra-info",
    type=str,
    default=None,
    help='Optional JSON-like string (e.g., \'{"tag":"test"}\') to store with the report.',
)
def evaluate(
    heuristic: str,
    num_games: int,
    seed: int,
    report_file: str,
    evaluator_board_width: int,
    evaluator_board_height: int,
    evaluator_max_frames: int,
    evaluator_process_pool: bool,
    extra_info: str | None,
) -> None:
    """Evaluate a specific Heuristic by playing multiple games."""
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pool_context: ProcessPoolExecutor | contextlib.nullcontext = (
        ProcessPoolExecutor() if evaluator_process_pool else contextlib.nullcontext()
    )

    try:
        process_pool_instance = pool_context.__enter__() if isinstance(pool_context, ProcessPoolExecutor) else None

        evaluator_instance = EvaluatorImpl(
            board_size=(evaluator_board_height, evaluator_board_width),
            max_evaluation_frames=evaluator_max_frames,
            process_pool=process_pool_instance,
        )

        detailed_evaluator = DetailedHeuristicEvaluator(
            num_games=num_games,
            seed=seed,  # Master seed for the evaluation setup
            evaluator=evaluator_instance,
            report_file=report_path,
        )

        heuristic_instance = _literal_eval_heuristic(heuristic)

        parsed_extra_info = None
        if extra_info:
            try:
                # Safely evaluate the extra_info string as a Python literal
                parsed_extra_info = ast.literal_eval(extra_info)
                if not isinstance(parsed_extra_info, dict):
                    # We expect a dict generally, but allow other simple types
                    pass
            except (ValueError, SyntaxError) as e:
                msg = f"Invalid format for --extra-info: '{extra_info}'. Must be JSON-like literal. Error: {e}"
                raise click.BadParameter(msg)

        click.echo(f"Evaluating heuristic: {heuristic_instance}")
        detailed_evaluator.evaluate(heuristic_instance, extra_info=parsed_extra_info)
        click.echo(f"Evaluation complete. Results appended to '{report_path}'.")

    except Exception:
        LOGGER.exception("An error occurred during evaluation.")
        traceback.print_exc()
    finally:
        if isinstance(pool_context, ProcessPoolExecutor) and hasattr(pool_context, "_shutdown"):
            click.echo("Shutting down process pool...")
            pool_context.shutdown(wait=True)
        click.echo("Evaluation process finished.")


@main.command()
@click.option("--width", type=int, required=True, help="Width of the space to fill.")
@click.option("--height", type=int, required=True, help="Height of the space to fill.")
@click.option("--holes", type=str, default=None, help='Define holes as "y1,x1,y2,x2;y1,x1,y2,x2;...".')
@click.option(
    "--inverted", is_flag=True, default=False, help="Invert the space (fill around holes instead of inside them)."
)
@click.option("--seed", type=int, default=None, help="Seed for random number generation during filling/coloring.")
@click.option("--use-rng/--no-rng", default=True, show_default=True, help="Use randomness in filling/coloring.")
@click.option(
    "--top-left-tendency/--no-top-left-tendency", default=True, show_default=True, help="Bias filler towards top-left."
)
@click.option(
    "--colorize/--no-colorize", default=True, show_default=True, help="Color the filled space using 4-coloring."
)
@click.option(
    "--minimum-separation-steps",
    type=int,
    default=0,
    show_default=True,
    help="Minimum steps between filling and coloring (for --colorize).",
)
@click.option(
    "--allow-coloring-retry/--no-allow-coloring-retry",
    default=True,
    show_default=True,
    help="Allow colorizer to retry once if it fails (for --colorize).",
)
@click.option(
    "--draw/--no-draw",
    default=True,
    show_default=True,
    help="Draw the filling/coloring process iteratively in the terminal.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    default=None,
    help="Save the final filled (and colored) numpy array(s) to .npy file(s).",
)
@click.option("--fuzz-test", is_flag=True, default=False, help="Run the dedicated space filling fuzz tester.")
def fill_space(
    width: int,
    height: int,
    holes: str | None,
    inverted: bool,
    seed: int | None,
    use_rng: bool,
    top_left_tendency: bool,
    colorize: bool,
    minimum_separation_steps: int,
    allow_coloring_retry: bool,
    draw: bool,
    output_file: str | None,
    fuzz_test: bool,
) -> None:
    """Fill (and optionally color) a 2D space with tetrominos."""
    if fuzz_test:
        click.echo("Starting Space Filling Fuzz Test Mode...")
        click.echo("This will run indefinitely with random parameters. Press Ctrl+C to stop.")
        try:
            # Call the fuzz test function directly
            fill_fuzz_test(draw=draw)  # Pass draw option to the fuzz tester
        except KeyboardInterrupt:
            click.echo("\nFuzz testing stopped.")
        except Exception:
            LOGGER.exception("An error occurred during fuzz testing.")
            traceback.print_exc()
        return  # Exit after fuzz test

    # --- Normal Operation ---
    try:
        hole_list = _parse_holes(holes)
        initial_space_bool = _generate_space(width, height, hole_list, inverted)
        initial_space_int = initial_space_bool.astype(np.int32) - 1  # -1 for holes, 0 for fillable

        if not TetrominoSpaceFiller.space_can_be_filled(initial_space_int):
            msg = (
                "The generated space configuration cannot be filled with tetrominos (island sizes not divisible by 4)."
            )
            raise click.ClickException(msg)

        final_filled_space: NDArray[np.int32] | None = None
        final_colored_space: NDArray[np.uint8] | None = None

        click.echo("Starting space filling...")
        start_time = time.monotonic()

        # Prepare drawing context
        draw_context = drawer.managed_draw_context() if draw else contextlib.nullcontext()

        with draw_context:
            if colorize:
                try:
                    color_generator = fill_and_colorize(
                        initial_space_bool,
                        use_rng=use_rng,
                        rng_seed=seed,
                        minimum_separation_steps=minimum_separation_steps,
                        allow_coloring_retry=allow_coloring_retry,
                    )
                    for filled_space, colored_space in color_generator:
                        if draw:
                            # Draw combined view: color where available, else filled block index
                            draw_array = np.where(colored_space > 0, colored_space, filled_space)
                            drawer.draw_array_fancy(draw_array)  # Uses internal sleep
                        final_filled_space = filled_space
                        final_colored_space = colored_space
                    # Get final state after StopIteration (returned value)
                    final_filled_space, final_colored_space = color_generator.value  # type: ignore

                except (NotFillableError, UnableToColorizeError) as e:
                    msg = f"Failed: {e}"
                    raise click.ClickException(msg)
            else:
                # Manual iteration with TetrominoSpaceFiller
                space_filler = TetrominoSpaceFiller(
                    initial_space_int,  # Operates inplace
                    use_rng=use_rng,
                    rng_seed=seed,
                    top_left_tendency=top_left_tendency,
                    space_updated_callback=partial(drawer.draw_array_fancy, initial_space_int) if draw else None,
                )
                try:
                    # If not drawing iteratively via callback, use the iterator
                    if draw and space_filler._space_updated_callback is None:  # noqa: SLF001
                        fill_iterator = space_filler.ifill()
                        for _ in fill_iterator:
                            drawer.draw_array_fancy(space_filler.space)
                            # pass # Yield happens implicitly, drawing done in loop
                    else:
                        # Either drawing via callback or not drawing at all
                        space_filler.fill()

                    final_filled_space = space_filler.space
                except NotFillableError as e:
                    msg = f"Failed: {e}"
                    raise click.ClickException(msg)

        end_time = time.monotonic()
        click.echo(f"Filling complete in {end_time - start_time:.2f} seconds.")

        # --- Validation and Output ---
        if final_filled_space is None:
            # Should not happen if no exception occurred, but check anyway
            click.echo("Warning: Final filled space not available.", err=True)
            return

        try:
            click.echo("Validating filled space...")
            TetrominoSpaceFiller.validate_filled_space(final_filled_space)
            if colorize and final_colored_space is not None:
                click.echo("Validating colored space...")
                FourColorizer.validate_colored_space(final_colored_space, final_filled_space)
            click.echo("Validation successful.")
        except ValueError as e:
            click.echo(f"Validation Failed: {e}", err=True)
            # Optionally draw the invalid final state if not drawn iteratively
            if not draw and final_filled_space is not None:
                draw_array = (
                    np.where(final_colored_space > 0, final_colored_space, final_filled_space)
                    if colorize and final_colored_space is not None
                    else final_filled_space
                )
                print("\n--- Final State (Validation Failed) ---")
                drawer.draw_full_array_raw(draw_array)  # Raw print might be better for errors

        if output_file:
            out_path = Path(output_file)
            try:
                if colorize and final_colored_space is not None:
                    # Save both filled and colored
                    filled_path = out_path.with_suffix(".filled.npy")
                    colored_path = out_path.with_suffix(".colored.npy")
                    np.save(filled_path, final_filled_space)
                    np.save(colored_path, final_colored_space)
                    click.echo(f"Saved filled space to '{filled_path}'")
                    click.echo(f"Saved colored space to '{colored_path}'")
                else:
                    # Save only filled
                    np.save(out_path.with_suffix(".npy"), final_filled_space)
                    click.echo(f"Saved filled space to '{out_path.with_suffix('.npy')}'")
            except Exception as e:
                click.echo(f"Error saving output file(s): {e}", err=True)

    except Exception:
        LOGGER.exception("An error occurred during space filling/coloring.")
        traceback.print_exc()
        click.echo("Process finished with errors.")


# --- Main Execution ---
if __name__ == "__main__":
    main()
