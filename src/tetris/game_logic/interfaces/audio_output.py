from pathlib import Path
from typing import Protocol


class AudioOutput(Protocol):
    def play_once(self, sound_file: str | Path, volume: float = 1) -> None:
        """Play the sound in the provided file once, non-blocking. Immediately return."""

    def play_on_loop(self, sound_file: str | Path, volume: float = 1) -> None:
        """Play the sound in the provided file in an endless loop, non-blocking. Immediately return."""

    def stop(self) -> None:
        """Stop all currently playing sounds."""
