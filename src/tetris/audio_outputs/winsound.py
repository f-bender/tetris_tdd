import logging
import winsound
from pathlib import Path

from tetris.game_logic.interfaces.audio_output import AudioOutput

_LOGGER = logging.getLogger(__name__)


class WinsoundAudioOutput(AudioOutput):
    """An audio output that uses the winsound module to play sounds. Only supports WAV files, and no volume control!"""

    def play_once(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file, volume)

        winsound.PlaySound(str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NOSTOP)

    def play_on_loop(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file, volume)

        winsound.PlaySound(
            str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NOSTOP | winsound.SND_LOOP
        )

    def stop(self) -> None:
        """Stop any currently playing sound."""
        winsound.PlaySound(None, winsound.SND_PURGE)

    def _check_input(self, sound_file: str | Path, volume: float) -> None:
        if not Path(sound_file).is_file():
            msg = f"Sound file {sound_file} does not exist."
            raise FileNotFoundError(msg)

        if Path(sound_file).suffix.lower() != ".wav":
            msg = f"WinsoundAudioOutput only supports WAV files. Got {Path(sound_file).suffix}."
            raise ValueError(msg)

        if volume != 1:
            _LOGGER.warning("Volume control is not supported. Ignoring volume parameter.")
