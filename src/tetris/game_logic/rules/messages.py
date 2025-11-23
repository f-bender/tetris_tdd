from enum import Enum, auto
from typing import NamedTuple

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board


class StartingLineClearMessage(NamedTuple):
    cleared_lines: list[int]
    num_frames: int


class FinishedLineClearMessage(NamedTuple):
    cleared_lines: list[int]


class Speed(Enum):
    NORMAL = auto()
    QUICK = auto()
    INSTANT = auto()


class StartMergeMessage(NamedTuple):
    speed: Speed
    duration: int


class FinishedMergeMessage(NamedTuple):
    pass


class WaitingForSpawnMessage(NamedTuple):
    game_index: int


class InstantSpawnMessage(NamedTuple):
    board: Board


class SpawnMessage(NamedTuple):
    block: Block
    next_block: Block


class SynchronizedSpawnCommandMessage(NamedTuple):
    pass


class BoardTranslationMessage(NamedTuple):
    x_offset: int = 0
    y_offset: int = 0


class Direction(Enum):
    LEFT = auto()
    RIGHT = auto()


class MoveMessage(NamedTuple):
    direction: Direction


class RotateMessage(NamedTuple):
    direction: Direction


class Tetris99Message(NamedTuple):
    num_lines: int
    target_id: int


class NumClearedLinesMessage(NamedTuple):
    num_cleared_lines: int
    session_max_cleared_lines: int


class NewLevelMessage(NamedTuple):
    level: int


class ScoreMessage(NamedTuple):
    score: int
    session_high_score: int


class PowerupTTLsMessage(NamedTuple):
    powerup_ttls: dict[int, int]  # mapping from powerup ID to TTL in frames
