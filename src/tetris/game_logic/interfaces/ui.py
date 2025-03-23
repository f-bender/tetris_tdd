from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block


@dataclass
class SingleUiElements:
    """Elements to be drawn on the UI for one game."""

    board: NDArray[np.uint8]
    next_block: Block | None = None
    score: int = 0
    # potential additions in the future: hold_block, level

    def reset(self) -> None:
        self.next_block = None
        self.score = 0


@dataclass
class UiElements:
    """All elements to be drawn on the UI."""

    games: Sequence[SingleUiElements]
    # global UI elements to be added in the future (e.g. pause menu)


class UI(ABC):
    @abstractmethod
    def initialize(self, board_height: int, board_width: int, num_boards: int) -> None: ...

    @abstractmethod
    def draw(self, elements: UiElements) -> None: ...

    def advance_startup(self) -> bool:
        """Advance the startup of the UI by one step. Return True if startup is finished."""
        return True
