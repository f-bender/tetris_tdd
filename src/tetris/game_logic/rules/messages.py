from enum import Enum, auto
from typing import NamedTuple

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board


class StartingLineFillMessage(NamedTuple):
    filled_lines: list[int]
    is_line_clear: bool
    num_frames: int


class FinishedLineFillMessage(NamedTuple):
    filled_lines: list[int]
    is_line_clear: bool


class Speed(Enum):
    NORMAL = auto()
    QUICK = auto()
    INSTANT = auto()


class MergeMessage(NamedTuple):
    speed: Speed


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
    rank: int
    session_high_score: int
    game_index: int


class PowerupTTLsMessage(NamedTuple):
    powerup_ttls: dict[int, int]  # mapping from powerup ID to TTL in frames


class PowerupTriggeredMessage(NamedTuple):
    position: tuple[int, int]


class PostMergeFinishedMessage(NamedTuple):
    pass


class GravityEffectTrigger(NamedTuple):
    per_col_probability: float = 1


class BotAssistanceStart(NamedTuple):
    pass


class BotAssistanceEnd(NamedTuple):
    pass


class ControllerSymbolUpdatedMessage(NamedTuple):
    controller_symbol: str


class FillLinesEffectTrigger(NamedTuple):
    num_lines: int


class GravityFinishedMessage(NamedTuple):
    pass


class GravityStartedMessage(NamedTuple):
    pass


class BlooperOverlayTrigger(NamedTuple):
    pass


class Tetris99FromPowerup(NamedTuple):
    num_lines: int
