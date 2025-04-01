# commands/helpers.py
import contextlib
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

# --- Assuming your project structure allows these imports ---
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.controllers.keyboard.pynput import PynputKeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.controller import Controller
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.game_logic.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.game_logic.rules.monitoring.track_score_rule import TrackScoreRule
from tetris.game_logic.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.game_logic.rules.special.parry_rule import ParryRule

# Import GamepadController if available
try:
    from inputs import devices

    from tetris.controllers.gamepad import GamepadController

    GAMEPAD_AVAILABLE = True
except ImportError:
    GamepadController = None  # type: ignore
    GAMEPAD_AVAILABLE = False

LOGGER = logging.getLogger(__name__)
DEFAULT_HEURISTIC = Heuristic()  # Use the default heuristic globally


def get_available_controller_specs() -> list[str]:
    """Returns a list of available controller spec strings."""
    specs = ["keyboard", "wasd", "vim", "bot"]
    if GAMEPAD_AVAILABLE:
        specs.append("gamepad")
    return specs


def parse_and_create_controllers(
    controller_specs: tuple[str, ...],
    num_games: int,
    boards: list[Board],
    use_process_pool: bool,
    ensure_consistent: bool,
    process_pool_instance: ProcessPoolExecutor | None,
) -> list[Controller]:
    """Parses controller specifications, creates Controller instances,
    and handles default assignments.
    """
    controllers: list[Controller] = []
    specs = list(controller_specs)  # Make mutable

    available_gamepads = 0
    if GAMEPAD_AVAILABLE:
        available_gamepads = len(devices.gamepads)

    # --- Determine effective controller specs ---
    if not specs:
        # Default: keyboard for first, bots for rest
        specs.append("keyboard")
        if num_games > 1:
            specs.extend(["bot"] * (num_games - 1))
        LOGGER.info("No controllers specified, using defaults: %s", specs)
    elif len(specs) == 1 and num_games > 1:
        # Replicate single spec, handling gamepad uniqueness
        spec_to_replicate = specs[0]
        if spec_to_replicate == "gamepad" and GAMEPAD_AVAILABLE:
            if available_gamepads >= num_games:
                specs = ["gamepad"] * num_games
            else:
                specs = ["gamepad"] * available_gamepads + ["bot"] * (num_games - available_gamepads)
                LOGGER.warning(
                    "Not enough gamepads (%d) for %d games. Using bots for the remainder.",
                    available_gamepads,
                    num_games,
                )
        else:
            specs = [spec_to_replicate] * num_games
        LOGGER.info("Single controller specified, replicating for all games: %s", specs)
    elif len(specs) != num_games:
        # This case should ideally be caught by click's nargs validation if possible,
        # but we double-check here.
        msg = (
            f"Number of controllers specified ({len(specs)}) must match number of games ({num_games})."
            f" Provide {num_games} controller types separated by spaces, or provide one type."
        )
        LOGGER.error(msg)
        raise ValueError(msg)  # Or handle more gracefully depending on desired CLI behavior

    # --- Create controller instances ---
    gamepad_idx_counter = 0
    for i, spec in enumerate(specs):
        board = boards[i]
        spec_lower = spec.lower()
        controller: Controller | None = None

        if spec_lower == "keyboard":
            controller = PynputKeyboardController.arrow_keys()
        elif spec_lower == "wasd":
            controller = PynputKeyboardController.wasd()
        elif spec_lower == "vim":
            controller = PynputKeyboardController.vim()
        elif spec_lower == "bot":
            # NOTE: Always use default heuristic, never lightning mode
            controller = HeuristicBotController(
                board,
                heuristic=DEFAULT_HEURISTIC,
                lightning_mode=False,  # Explicitly False
                process_pool=process_pool_instance,
                ensure_consistent_behaviour=ensure_consistent,
            )
        elif spec_lower == "gamepad":
            if not GAMEPAD_AVAILABLE:
                LOGGER.error("Gamepad support not available but 'gamepad' controller requested.")
                msg = "Gamepad controller requested but support is unavailable."
                raise ValueError(msg)
            if gamepad_idx_counter >= available_gamepads:
                LOGGER.error(
                    "Requested gamepad controller %d, but only %d found.", gamepad_idx_counter + 1, available_gamepads
                )
                msg = "Not enough gamepads available."
                raise ValueError(msg)
            controller = GamepadController(gamepad_index=gamepad_idx_counter)
            gamepad_idx_counter += 1
        else:
            LOGGER.error("Unknown controller type: '%s'. Choices: %s", spec, get_available_controller_specs())
            msg = f"Unknown controller type: '{spec}'"
            raise ValueError(msg)

        # Ensure game_index is set for relevant controllers (needed for dependency wiring)
        # HeuristicBotController manages its index via the board association implicitly handled by DEPENDENCY_MANAGER
        if isinstance(controller, Callback | Subscriber | Publisher) and not isinstance(
            controller, HeuristicBotController
        ):
            # This check might be overly cautious if no other controllers implement these
            # but it's safer. HeuristicBotController handles its own index lookup.
            if hasattr(controller, "game_index"):
                controller.game_index = i
                LOGGER.debug("Set game_index %d for controller %s", i, type(controller).__name__)
            else:
                LOGGER.warning(
                    "Controller %s might need game_index, but attribute not found.", type(controller).__name__
                )

        controllers.append(controller)
        LOGGER.debug("Created controller %d: %s", i, type(controller).__name__)

    return controllers


def create_rules_and_callbacks_for_game(
    game_index: int, num_games: int, spawn_delay: int, seed: int | None, use_tetris_99_rules: bool
) -> RuleSequence:
    """Creates rules and registers callbacks for a specific game instance."""
    DEPENDENCY_MANAGER.current_game_index = game_index
    LOGGER.debug("Setting up rules and callbacks for game_index %d", game_index)

    # Register callbacks (they add themselves to DEPENDENCY_MANAGER)
    TrackScoreRule(header=f"Game {game_index}")

    # Create Spawn Strategy with seed if provided
    if seed is not None:
        game_seed = seed + game_index  # Simple way to get different seeds per game
        spawn_strategy = SpawnStrategyImpl.from_shuffled_bag(seed=game_seed)
        LOGGER.debug("Using seeded SpawnStrategy (seed %d) for game %d", game_seed, game_index)
    else:
        spawn_strategy = SpawnStrategyImpl()  # Use default random
        LOGGER.debug("Using default random SpawnStrategy for game %d", game_index)

    rules: list[Any] = [
        MoveRule(),
        RotateRule(),
        SpawnDropMergeRule(spawn_delay=spawn_delay, spawn_strategy=spawn_strategy),
        ParryRule(),
    ]

    if use_tetris_99_rules and num_games > 1:
        target_idxs = list(set(range(num_games)) - {game_index})
        rules.append(Tetris99Rule(target_idxs=target_idxs))
        LOGGER.debug("Added Tetris99Rule for game %d targeting %s", game_index, target_idxs)

    return RuleSequence(rules)


def parse_holes(holes_str: str | None) -> list[tuple[int, int, int, int]]:
    """Parses a string like "y1,x1,y2,x2;y1,x1,y2,x2" into a list of tuples."""
    if not holes_str:
        return []
    holes = []
    try:
        hole_definitions = holes_str.split(";")
        for hole_def in hole_definitions:
            if not hole_def.strip():
                continue  # Skip empty parts
            parts = [int(p.strip()) for p in hole_def.split(",")]
            if len(parts) != 4:
                msg = "Each hole definition must have 4 comma-separated integers."
                raise ValueError(msg)
            y1, x1, y2, x2 = parts
            if not (0 <= y1 < y2 and 0 <= x1 < x2):
                msg = f"Invalid hole coordinates (must be y1 < y2 and x1 < x2): {parts}"
                raise ValueError(msg)
            holes.append((y1, x1, y2, x2))
        LOGGER.debug("Parsed holes: %s", holes)
        return holes
    except ValueError as e:
        LOGGER.exception("Invalid format for holes string '%s': %s", holes_str, e)
        msg = f"Invalid format for holes: '{holes_str}'. Error: {e}. Expected: 'y1,x1,y2,x2;y1,x1,y2,x2;...'."
        raise ValueError(msg) from e


def generate_space(
    width: int, height: int, holes: list[tuple[int, int, int, int]], inverted: bool
) -> NDArray[np.bool_]:
    """Generates the initial boolean space array."""
    if not (width > 0 and height > 0):
        LOGGER.error("Space width (%d) and height (%d) must be positive.", width, height)
        msg = "Width and height must be positive."
        raise ValueError(msg)

    space = np.ones((height, width), dtype=bool)
    LOGGER.debug("Created initial space of size %dx%d", height, width)

    for y1, x1, y2, x2 in holes:
        if not (y2 <= height and x2 <= width):
            LOGGER.error(
                "Hole coordinates %d,%d,%d,%d are outside the space dimensions %dx%d.", y1, x1, y2, x2, height, width
            )
            msg = f"Hole coordinates {y1},{x1},{y2},{x2} are outside the space dimensions {height}x{width}."
            raise ValueError(msg)
        space[y1:y2, x1:x2] = False
        LOGGER.debug("Applied hole from (%d, %d) to (%d, %d)", y1, x1, y2, x2)

    if inverted:
        space = ~space
        LOGGER.debug("Inverted space.")

    return space


def manage_process_pool(use_pool: bool) -> contextlib.AbstractContextManager[ProcessPoolExecutor | None]:
    """Returns a context manager for the ProcessPoolExecutor if requested."""
    if use_pool:
        LOGGER.info("Using ProcessPoolExecutor.")
        # max_workers defaults to number of processors, which is usually reasonable
        return ProcessPoolExecutor()
    LOGGER.info("Not using ProcessPoolExecutor.")
    return contextlib.nullcontext(None)
