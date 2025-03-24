from abc import ABC
from dataclasses import dataclass


@dataclass(kw_only=True)
class AnimationSpec(ABC):
    current_frame: int = -1
    total_frames: int

    def advance_frame(self) -> None:
        self.current_frame += 1

    @property
    def done(self) -> bool:
        return self.current_frame >= self.total_frames


@dataclass(kw_only=True)
class TetrisAnimationSpec(AnimationSpec):
    top_line_idx: int


@dataclass(kw_only=True)
class LineClearAnimationSpec(AnimationSpec):
    cleared_lines: list[int]
