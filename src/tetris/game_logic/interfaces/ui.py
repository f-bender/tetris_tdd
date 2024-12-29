from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


class UI(ABC):
    @abstractmethod
    def initialize(self, board_height: int, board_width: int, num_boards: int = 1) -> None: ...

    @abstractmethod
    def draw(self, boards: Iterable[NDArray[np.uint8]]) -> None: ...

    def advance_startup(self) -> bool:
        """Advance the startup of the UI by one step. Return True if startup is finished."""
        return True
