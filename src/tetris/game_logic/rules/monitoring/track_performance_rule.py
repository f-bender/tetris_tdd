import time

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.dependency_manager import DependencyManager


class TrackPerformanceCallback(Callback):
    BAR_WIDTH = 100

    def __init__(self, fps: float = 60) -> None:
        super().__init__()

        self._last_frame_start: float | None = None
        self._frame_budget = 1 / fps

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == DependencyManager.RUNTIME_INDEX

    def on_frame_start(self, game_index: int) -> None:
        self._last_frame_start = time.perf_counter()

    def on_frame_end(self, game_index: int) -> None:
        if not self._last_frame_start:
            return

        frame_time = time.perf_counter() - self._last_frame_start
        budget_ratio = frame_time / self._frame_budget

        if budget_ratio >= 1:
            full_blocks = self.BAR_WIDTH - 1
            partial_block_ratio = 1.0
        else:
            _full_blocks, partial_block_ratio = divmod(budget_ratio * self.BAR_WIDTH, 1)
            full_blocks = int(_full_blocks)

        partial_block_eighths = round(partial_block_ratio * 8)
        partial_block_code = (
            ord(" ")
            if partial_block_eighths == 0
            else (
                0x258F  # one eighth block
                - (partial_block_eighths - 1)
            )
        )

        print(  # noqa: T201
            "Frame budget usage: "
            f"|{'\u2588' * full_blocks}{chr(partial_block_code)}{' ' * (self.BAR_WIDTH - full_blocks - 1)}|"
        )
