import logging
import multiprocessing
import random
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from queue import Empty
from typing import Any

# Import your project modules:
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic
from tetris.game_logic.components import Board
from tetris.game_logic.components.block import Block
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import UI
from tetris.heuristic_bot_gym.heuristic_bot_gym import Evaluator
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategyImpl
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.ui.cli.ui import CLI

LOGGER = logging.getLogger(__name__)

# Global variable for the communication queue.
_GLOBAL_COMM_QUEUE = None


def _init_worker(comm_queue: multiprocessing.Queue) -> None:
    """This initializer is run once in each worker process.
    It sets a global variable so that workers can send board updates.
    """
    global _GLOBAL_COMM_QUEUE
    _GLOBAL_COMM_QUEUE = comm_queue


def _evaluate_task(
    slot_id: int,
    heuristic: Heuristic,
    board_size: tuple[int, int],
    max_evaluation_frames: int,
    block_selection_fn_from_seed: Callable[[int], Callable[[], Block]],
    seed: int,
) -> float:
    """Worker function to evaluate one heuristic.
    Uses the global _GLOBAL_COMM_QUEUE (set via the initializer) to send board updates.
    """
    DEPENDENCY_MANAGER.current_game_index = slot_id
    board = Board.create_empty(*board_size)
    controller = HeuristicBotController(board, lightning_mode=True)
    controller.heuristic = heuristic
    score_tracker = TrackScoreCallback()
    spawn_strategy = SpawnStrategyImpl()
    spawn_strategy.select_block_fn = block_selection_fn_from_seed(seed)
    game = Game(
        board=board,
        controller=controller,
        rule_sequence=RuleSequence(
            (
                MoveRule(),
                RotateRule(),
                SpawnDropMergeRule(
                    spawn_strategy=spawn_strategy,
                    speed_strategy=SpeedStrategyImpl(base_interval=10),
                    spawn_delay=0,
                ),
                ClearFullLinesRule(),
            )
        ),
        callback_collection=CallbackCollection((score_tracker,)),
    )
    DEPENDENCY_MANAGER.wire_up()

    # Send the initial board state.
    try:
        _GLOBAL_COMM_QUEUE.put(("update", slot_id, board.as_array()))
    except Exception as exc:
        LOGGER.exception("Error sending initial update for slot %d: %s", slot_id, exc)

    for _ in range(max_evaluation_frames):
        try:
            game.advance_frame()
        except GameOverError:
            break
        try:
            _GLOBAL_COMM_QUEUE.put(("update", slot_id, board.as_array()))
        except Exception as exc:
            LOGGER.exception("Error sending update for slot %d: %s", slot_id, exc)
    return score_tracker.score


class ParallelEvaluator(Evaluator):
    """Evaluates a list of heuristics by running up to N evaluations in parallel,
    where N = min(number of CPUs, number of heuristics). As soon as one evaluation
    finishes, the next heuristic is scheduled in its place. Board updates are sent
    via a shared queue (set up via the initializer) to the main process, which drives
    the UI.
    """

    def __init__(
        self,
        board_size: tuple[int, int] = (20, 10),
        max_evaluation_frames: int = 100_000,
        block_selection_fn_from_seed: Callable[[int], Callable[[], Block]] = SpawnStrategyImpl.truly_random_select_fn,
        ui_class: type[UI] | None = CLI,
    ) -> None:
        self._ui = None if ui_class is None else ui_class()
        self._board_size = board_size
        self._max_frames = max_evaluation_frames
        self._block_selection_fn_from_seed = block_selection_fn_from_seed
        self._initialized = False

    @property
    def config(self) -> dict[str, Any]:
        return {
            "board_size": self._board_size,
            "max_evaluation_frames": self._max_frames,
            "block_selection_fn_from_seed": self._block_selection_fn_from_seed,
            "ui_class": None if self._ui is None else type(self._ui),
        }

    def _initialize(self, num_slots: int) -> None:
        self._initialized = True
        # Holds the latest board state for each slot.
        self._boards = [None] * num_slots
        if self._ui:
            self._ui.initialize(*self._board_size, num_boards=num_slots)

    def __call__(self, heuristics: list[Heuristic]) -> list[float]:
        total = len(heuristics)
        n_slots = min(multiprocessing.cpu_count(), total)
        if not self._initialized:
            self._initialize(n_slots)
        final_scores = [None] * total

        # Create a shared queue for board updates.
        comm_queue = multiprocessing.Queue()
        seed = random.randrange(2**32)

        # Create a ProcessPoolExecutor with an initializer that sets the global queue.
        with ProcessPoolExecutor(max_workers=n_slots, initializer=_init_worker, initargs=(comm_queue,)) as executor:
            next_index = 0
            # Map each future to (slot_id, global index)
            future_to_slot: dict[Any, tuple[int, int]] = {}
            for slot_id in range(n_slots):
                if next_index < total:
                    heuristic = heuristics[next_index]
                    future = executor.submit(
                        _evaluate_task,
                        slot_id,
                        heuristic,
                        self._board_size,
                        self._max_frames,
                        self._block_selection_fn_from_seed,
                        seed,
                    )
                    future_to_slot[future] = (slot_id, next_index)
                    next_index += 1

            remaining = total
            # Main loop: poll for board updates and check completed tasks.
            while remaining > 0:
                try:
                    while True:  # Drain all available messages.
                        msg = comm_queue.get_nowait()
                        if msg[0] == "update":
                            slot_id, board_state = msg[1], msg[2]
                            self._boards[slot_id] = board_state
                except Empty:
                    pass

                if self._ui:
                    if isinstance(self._ui, CLI):
                        completed = total - remaining
                        print(f"Completed: {completed} / {total}")
                    self._ui.advance_startup()
                    self._ui.draw(self._boards)
                time.sleep(1 / 60)

                done_futures = [fut for fut in future_to_slot if fut.done()]
                for fut in done_futures:
                    slot_id, global_index = future_to_slot.pop(fut)
                    try:
                        score = fut.result()
                    except Exception as exc:
                        score = None
                        LOGGER.exception("Task for heuristic %d (slot %d) failed: %s", global_index, slot_id, exc)
                    final_scores[global_index] = score
                    remaining -= 1
                    # If there are more heuristics, schedule the next one in this slot.
                    if next_index < total:
                        heuristic = heuristics[next_index]
                        new_future = executor.submit(
                            _evaluate_task,
                            slot_id,
                            heuristic,
                            self._board_size,
                            self._max_frames,
                            self._block_selection_fn_from_seed,
                            seed,
                        )
                        future_to_slot[new_future] = (slot_id, next_index)
                        next_index += 1

        return final_scores
