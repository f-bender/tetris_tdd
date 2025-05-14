import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue

from playsound3 import playsound
from playsound3.playsound3 import Sound

from tetris.game_logic.interfaces.audio_output import AudioOutput

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Sound:
    sound: Sound
    loop: bool
    path: str | Path


class Playsound3AudioOutput(AudioOutput):
    """An audio output that uses the playsound3 module to play sounds. Doesn't support volume control!"""

    def __init__(self) -> None:
        self._playing_sounds: SimpleQueue[_Sound] = SimpleQueue()

        self._manager_thread = threading.Thread(target=self._manage_sounds, daemon=True)
        self._manager_thread.start()

    def _manage_sounds(self) -> None:
        # NOTE: this is a significant downer in terms of performance!
        while True:
            time.sleep(0.01)
            for _ in range(self._playing_sounds.qsize()):
                try:
                    sound = self._playing_sounds.get_nowait()
                except Empty:
                    break

                if sound.sound.is_alive():
                    self._playing_sounds.put_nowait(sound)
                    continue

                if sound.loop:
                    sound.sound = playsound(sound.path, block=False)
                    self._playing_sounds.put_nowait(sound)

    def play_once(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file, volume)

        self._playing_sounds.put_nowait(_Sound(sound=playsound(sound_file, block=False), loop=False, path=sound_file))

    def play_on_loop(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file, volume)

        self._playing_sounds.put_nowait(_Sound(sound=playsound(sound_file, block=False), loop=True, path=sound_file))

    def stop(self) -> None:
        """Stop any currently playing sound."""
        while True:
            try:
                self._playing_sounds.get_nowait().sound.stop()
            except Empty:
                break

    def _check_input(self, sound_file: str | Path, volume: float) -> None:
        if not str(sound_file).startswith("http") and not Path(sound_file).is_file():
            msg = f"Sound file {sound_file} does not exist."
            raise FileNotFoundError(msg)

        if volume != 1:
            _LOGGER.warning("Volume control is not supported. Ignoring volume parameter.")
