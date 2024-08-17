from typing import Any, Protocol


class Callback(Protocol):
    def custom_message(self, message: Any) -> None: ...
