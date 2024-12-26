from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class UI(ABC):
    @abstractmethod
    def initialize(self, board_height: int, board_width: int) -> None: ...

    @abstractmethod
    def draw(self, board: NDArray[np.bool]) -> None: ...

    def advance_startup(self) -> bool:
        """Advance the startup of the UI by one step. Return True if startup is finished."""
        return True
