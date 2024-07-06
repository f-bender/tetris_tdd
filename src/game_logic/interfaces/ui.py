from typing import Protocol

from game_logic.components import Board

# TODO draw should take the board string representation needed for drawing, not the whole board (need-to-know)


class UI(Protocol):
    def draw(self, board: Board) -> None: ...
    def initialize(self) -> None: ...
    def game_over(self, board: Board) -> None: ...  # TODO should also get a score
