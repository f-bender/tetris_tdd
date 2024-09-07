import io
import sys
from types import TracebackType
from typing import Self


class BufferedPrint:
    """Redirection of stdout to a buffer which is printed to the console all at once at the desired time."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()

    def is_active(self) -> bool:
        return sys.stdout is self._buffer

    def start_buffering(self) -> Self:
        if self.is_active():
            raise RuntimeError("BufferedPrint is already active")

        sys.stdout = self._buffer
        return self

    def print_and_reset_buffer(self) -> None:
        if not self.is_active():
            raise RuntimeError("BufferedPrint is not active")

        sys.stdout = sys.__stdout__
        print(self._buffer.getvalue(), end="")
        self._buffer.close()
        self._buffer = io.StringIO()

    def discard_and_reset_buffer(self) -> None:
        if not self.is_active():
            raise RuntimeError("BufferedPrint is not active")

        sys.stdout = sys.__stdout__
        self._buffer.close()
        self._buffer = io.StringIO()

    def print_and_restart_buffering(self) -> None:
        self.print_and_reset_buffer()
        self.start_buffering()

    def __enter__(self) -> Self:
        return self.start_buffering()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.print_and_reset_buffer()
