import os
from game_logic.components import Board
from colorama import Back

CURSOR_UP = "\033[1A"
CLEAR = "\x1b[2K"


class CLI:
    def initialize(self) -> None:
        os.system("cls")

    def draw(self, board: Board) -> None:
        self._draw_board(board)
        self._reset_cursor_after_drawing_board(board)

    def _draw_board(self, board: Board) -> None:
        print(
            str(board).replace(".", f"{Back.LIGHTBLACK_EX}  {Back.RESET}").replace("X", f"{Back.WHITE}  {Back.RESET}")
        )

    def _reset_cursor_after_drawing_board(self, board: Board) -> None:
        print(CURSOR_UP * (board.height + 1) + CLEAR, end="")

    def game_over(self, board: Board) -> None:
        self._draw_board(board)
        print("Game Over")
