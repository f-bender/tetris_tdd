from typing import Protocol

import numpy as np

from numpy.typing import NDArray


class UI(Protocol):
    def initialize(self, board_height: int, board_width: int) -> None: ...
    def draw(self, board: NDArray[np.bool]) -> None: ...
    def game_over(self, board: NDArray[np.bool]) -> None: ...
