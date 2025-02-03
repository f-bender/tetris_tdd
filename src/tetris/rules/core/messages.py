from enum import Enum, auto
from typing import NamedTuple

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board


class LineClearMessage(NamedTuple):
    cleared_lines: list[int]


class Speed(Enum):
    NORMAL = auto()
    QUICK = auto()
    INSTANT = auto()


class MergeMessage(NamedTuple):
    speed: Speed


class InstantSpawnMessage(NamedTuple):
    board: Board


class SpawnMessage(NamedTuple):
    block: Block
    next_block: Block
