from abc import ABC, abstractmethod

from tetris.game_logic.components.board import Board


class GradualBoardManipulation(ABC):
    @abstractmethod
    def manipulate_gradually(self, board: Board, current_frame: int, total_frames: int) -> None:
        """Gradually apply a board manipulation to the board, over a specified number of frames.

        Args:
            board: The board to manipulate.
            current_frame: The current frame of the manipulation. Must be in the range from 0 to total_frames - 1.
            total_frames: The total number of frames to apply the manipulation over.
        """

    def done_already(self) -> bool:
        return False


class BoardManipulation(GradualBoardManipulation):
    def manipulate_gradually(self, board: Board, current_frame: int, total_frames: int) -> None:
        if current_frame == 0:
            self.manipulate(board)

    @abstractmethod
    def manipulate(self, board: Board) -> None: ...

    def done_already(self) -> bool:
        return True
