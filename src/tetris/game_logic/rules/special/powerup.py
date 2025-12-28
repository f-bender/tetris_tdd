import logging
import random
from itertools import chain
from typing import NamedTuple, override

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.core.spawn.spawn import SpawnRule
from tetris.game_logic.rules.messages import PowerupTriggeredMessage, PowerupTTLsMessage, SpawnMessage
from tetris.game_logic.rules.special.powerup_effect import PowerupEffectManager

_LOGGER = logging.getLogger(__name__)


# Note: Protocols (like Rule) need to go last for MRO reasons
class PowerupRule(Publisher, Subscriber, Callback, Rule):
    POWERUP_SLOT_OFFSET = Board.MAX_REGULAR_CELL_VALUE + 1

    def __init__(
        self,
        *,
        powerup_spawn_probability: float = 0.04,
        # 10-20 seconds at 60 FPS
        min_ttl_frames: int = 600,
        max_ttl_frames: int = 1200,
    ) -> None:
        super().__init__()
        # note: np.uint8 is the dtype used by the board
        self._powerup_ttls = np.zeros(np.iinfo(np.uint8).max + 1, dtype=np.uint16)
        self._powerup_positions: dict[int, tuple[int, int]] = {}
        self._powerup_spawn_probability = powerup_spawn_probability

        self._min_ttl_frames = min_ttl_frames
        self._max_ttl_frames = max_ttl_frames

        self._powerup_effect_manager = PowerupEffectManager()

    @override
    def on_game_start(self, game_index: int) -> None:
        self._powerup_ttls[...] = 0
        self._powerup_positions.clear()
        self._powerup_effect_manager.reset()

    def put_powerup_on_hold(self, slot: int) -> None:
        """Put the power-up in the given slot on hold.

        Meaning:
        - it is should be no longer be on the board (responsibility of the caller)
        - its TTL counter is frozen (its slots remains occupied and can't be taken by a new powerup)

        Main envisioned use case: "held block" feature.
        """
        assert self._powerup_ttls[slot] > 0, "Trying to put a non-existing power-up on hold!"
        del self._powerup_positions[slot]

    @override
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        """Decrease the TTL of all power-ups by 1. Remove power-ups with TTL 0."""
        actually_present_powerup_slots = [
            i
            for i in chain(
                # note: need to get cells from board and from active block separately (instead of the "all-in-one"
                # as_array()) because the active block may be partly outside the board (on top)
                np.unique(board.array_view_without_active_block()),
                np.unique(board.active_block.block.cells) if board.active_block is not None else [],
            )
            if Board.MIN_POWERUP_CELL_VALUE <= i <= Board.MAX_POWERUP_CELL_VALUE
        ]

        # Update power-up positions
        board_array = board.as_array()
        for slot in actually_present_powerup_slots:
            positions = np.argwhere(board_array == slot)
            if len(positions) == 0:
                # edge case: the powerup is part of the active block and outside the board (on top) - skip updating its
                # position
                continue

            assert len(positions) == 1, f"Power-up slot {slot} present {len(positions)} times on the board!"
            self._powerup_positions[slot] = tuple(positions[0])

        # Decrease TTLs of present power-ups (TTLs which are > 0 but not present on the board are on hold)
        powerups_before = self._powerup_ttls > 0
        assert powerups_before[actually_present_powerup_slots].all(), "Board contains powerups without TTL!"
        self._powerup_ttls[actually_present_powerup_slots] -= 1
        powerups_after = self._powerup_ttls > 0

        # Remove powerups whose TTL has just reached 0
        just_decayed_powerups = np.where(powerups_before & ~powerups_after)[0]
        self._decay_powerups_in_board(board=board, just_decayed_powerups=just_decayed_powerups)

        # Clean up positions of decayed power-ups
        for slot in just_decayed_powerups:
            del self._powerup_positions[slot]

        actually_present_powerup_slots_set = set(actually_present_powerup_slots) - set(just_decayed_powerups)
        registered_powerup_slots = set(np.where(powerups_after)[0])
        assert actually_present_powerup_slots_set <= registered_powerup_slots

        # Notify subscribers about current power-up TTLs
        self.notify_subscribers(
            PowerupTTLsMessage(
                powerup_ttls={
                    int(powerup): int(self._powerup_ttls[powerup]) for powerup in actually_present_powerup_slots_set
                }
            )
        )

        # Trigger power-ups which have just been activated
        # (i.e. are registered, and not on hold, but are not present in the board this frame)
        just_triggered_powerups = registered_powerup_slots - actually_present_powerup_slots_set
        for slot in just_triggered_powerups:
            if slot not in self._powerup_positions:
                # the powerup in this slot is on hold, it isn't currently on the board
                # (but this doesn't mean it has just been triggered)
                continue

            self.notify_subscribers(PowerupTriggeredMessage(self._powerup_positions[slot]))
            del self._powerup_positions[slot]
            self._powerup_effect_manager.trigger_random_effect()
            self._powerup_ttls[slot] = 0

        self._powerup_effect_manager.apply(frame_counter, action_counter, board)

    def _decay_powerups_in_board(self, board: Board, just_decayed_powerups: NDArray[np.intp]) -> None:
        # note: ghost block doesn't need to be considered as it should never be present in the actual board (view), just
        # in the array generated by as_array(include_ghost=True)
        board_view = board.array_view_without_active_block()
        board_view[np.isin(board_view, just_decayed_powerups)] %= self.POWERUP_SLOT_OFFSET

        if board.active_block is not None:
            active_block_cells = board.active_block.block.cells
            active_block_cells[np.isin(active_block_cells, just_decayed_powerups)] %= self.POWERUP_SLOT_OFFSET

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, SpawnRule) and publisher.game_index == self.game_index

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        if not any(isinstance(p, SpawnRule) for p in publishers):
            msg = f"{type(self).__name__} of game {self.game_index} is not subscribed to a SpawnRule: {publishers}"
            raise RuntimeError(msg)

    @override
    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, SpawnMessage) and random.random() < self._powerup_spawn_probability:
            self._spawn_powerup(message.next_block)

    def _spawn_powerup(self, block: Block) -> None:
        used_powerup_slots = set(np.where(self._powerup_ttls > 0)[0])

        # note: we assume the block doesn't yet contain any power-up cells
        new_powerup_slot = np.max(block.cells) + self.POWERUP_SLOT_OFFSET
        while new_powerup_slot in used_powerup_slots:
            new_powerup_slot += self.POWERUP_SLOT_OFFSET
            if new_powerup_slot > Board.MAX_POWERUP_CELL_VALUE:
                _LOGGER.warning("No available power-up slots! Not spawning a power-up.")
                return

        powerup_position = random.choice(list(zip(*np.nonzero(block.cells), strict=True)))
        block.cells[powerup_position] = new_powerup_slot

        self._powerup_ttls[new_powerup_slot] = random.randint(self._min_ttl_frames, self._max_ttl_frames)

        self.notify_subscribers(
            PowerupTTLsMessage(
                powerup_ttls={
                    int(powerup): int(self._powerup_ttls[powerup]) for powerup in np.where(self._powerup_ttls > 0)[0]
                }
            )
        )
