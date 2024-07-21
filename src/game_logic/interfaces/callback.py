from typing import Protocol


class Callback(Protocol):
    def custom_message(self, message: str) -> None: ...
