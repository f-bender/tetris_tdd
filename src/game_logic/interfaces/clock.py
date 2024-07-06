from typing import Protocol


class Clock(Protocol):
    def tick(self) -> None: ...
