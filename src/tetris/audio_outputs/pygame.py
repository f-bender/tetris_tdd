from pathlib import Path

import pygame

from tetris.game_logic.interfaces.audio_output import AudioOutput


class PygameAudioOutput(AudioOutput):
    """An audio output that uses the winsound module to play sounds. Only supports WAV files, and no volume control!"""

    def __init__(self) -> None:
        pygame.mixer.init()

    def play_once(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file)

        sound = pygame.mixer.Sound(sound_file)
        sound.set_volume(volume)
        sound.play()

    def play_on_loop(self, sound_file: str | Path, volume: float = 1) -> None:
        self._check_input(sound_file)

        sound = pygame.mixer.Sound(sound_file)
        sound.set_volume(volume)
        sound.play(loops=-1)

    def stop(self) -> None:
        pygame.mixer.stop()

    def _check_input(self, sound_file: str | Path) -> None:
        if not Path(sound_file).is_file():
            msg = f"Sound file {sound_file} does not exist."
            raise FileNotFoundError(msg)
