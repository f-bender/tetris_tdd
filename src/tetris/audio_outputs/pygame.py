from functools import cache
from pathlib import Path

import pygame

from tetris.game_logic.interfaces.audio_output import AudioOutput

pygame.mixer.init()
# high number of channels to allow high number of overlapping sounds without any being dropped
pygame.mixer.set_num_channels(64)


@cache
def _get_sound(sound_file: str | Path, volume: float = 1) -> pygame.mixer.Sound:
    if not Path(sound_file).is_file():
        msg = f"Sound file {sound_file} does not exist."
        raise FileNotFoundError(msg)

    sound = pygame.mixer.Sound(sound_file)
    sound.set_volume(volume)
    return sound


class PygameAudioOutput(AudioOutput):
    """An audio output that uses pygame.mixer to play sounds."""

    def play_once(self, sound_file: str | Path, volume: float = 1) -> None:
        _get_sound(sound_file, volume).play()

    def play_on_loop(self, sound_file: str | Path, volume: float = 1) -> None:
        _get_sound(sound_file, volume).play(loops=-1)

    def stop(self) -> None:
        pygame.mixer.stop()
