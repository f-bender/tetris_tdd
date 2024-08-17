from typing import NamedTuple, Protocol


class Callback(Protocol):
    def custom_message(self, message: NamedTuple) -> None: ...
