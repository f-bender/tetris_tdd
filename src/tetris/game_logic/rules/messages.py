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


class MergeMessage(NamedTuple):
    speed: Speed
    next_spawn_frame_counter: int


class InstantSpawnMessage(NamedTuple):
    board: Board


class SpawnMessage(NamedTuple):
    block: Block
    next_block: Block


class BoardTranslationMessage(NamedTuple):
    x_offset: int = 0
    y_offset: int = 0
