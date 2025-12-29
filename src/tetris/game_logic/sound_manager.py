import asyncio
import random
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, override

import httpx

from tetris.game_logic.interfaces.audio_output import AudioOutput
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines
from tetris.game_logic.rules.board_manipulations.gravity import Gravity
from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.messages import (
    GravityFinishedMessage,
    GravityStartedMessage,
    MergeMessage,
    MoveMessage,
    NewLevelMessage,
    PowerupTriggeredMessage,
    RotateMessage,
    StartingLineFillMessage,
)
from tetris.game_logic.rules.special.powerup import PowerupRule


class Sound(StrEnum):
    MUSIC1 = "music1"
    MUSIC2 = "music2"
    MUSIC3 = "music3"
    MOVE = "move"
    ROTATE = "rotate"
    MERGE = "merge"
    LINE_CLEAR = "line_clear"
    TETRIS_CLEAR = "tetris_clear"
    NEXT_LEVEL = "next_level"
    GAME_OVER = "game_over"
    POWERUP = "powerup"
    LINE_FILL = "line_fill"
    GRAVITY = "gravity"


class SoundManager(Subscriber, Callback):
    _SOUND_PACKS_DIR = Path(__file__).parents[3] / "data" / "sounds"

    _ONLINE_SOUND_PACKS: Mapping[str, Mapping[Sound, str]] = MappingProxyType(
        {
            "tetris_nes": {
                Sound.MUSIC1: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/1%20-%20Music%201.mp3",
                Sound.MUSIC2: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/2%20-%20Music%202.mp3",
                Sound.MUSIC3: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/3%20-%20Music%203.mp3",
                Sound.MOVE: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%204.mp3",
                Sound.ROTATE: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%206.mp3",
                Sound.MERGE: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%208.mp3",
                Sound.LINE_CLEAR: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%2011.mp3",
                Sound.TETRIS_CLEAR: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%205.mp3",
                Sound.NEXT_LEVEL: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%207.mp3",
                Sound.GAME_OVER: "https://fi.zophar.net/soundfiles/nintendo-nes-nsf/tetris-1989-Nintendo/SFX%2014.mp3",
                # POWERUP: manually downloaded from https://pixabay.com/sound-effects/8-bit-powerup-6768/
                # LINE_FILL: LINE_CLEAR, but reversed and cut to contain only the last 0.5s
                # GRAVITY: downloaded from https://www.youtube.com/watch?v=__PiWa3CYJ0, sped up 9x
            }
        }
    )

    def __init__(
        self, audio_output: AudioOutput, sound_pack: str = "tetris_nes", game_indices: Collection[int] | None = None
    ) -> None:
        super().__init__()

        if game_indices is not None and len(game_indices) == 0:
            msg = (
                "`game_indices` should not be empty. If all games should have sound, pass `None`. "
                "If no games should have sound, don't create a SoundManager at all!"
            )
            raise ValueError(msg)

        self._audio_output = audio_output
        self._game_indices = game_indices

        sound_pack_dir = self._SOUND_PACKS_DIR / sound_pack

        if not sound_pack_dir.is_dir():
            if sound_pack in self._ONLINE_SOUND_PACKS:
                self._download(sound_pack)
            else:
                msg = f"Unknown sound pack '{sound_pack}'!"
                raise ValueError(msg)

        self._sound_files = {sound: next(sound_pack_dir.glob(f"{sound}.*"), None) for sound in Sound}

        self._enabled = True

        # this ignores enabled state; used to track whether music *should* be playing
        self._music_playing = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        if self._music_playing:
            self._start_playing_music()

    def disable(self) -> None:
        self._audio_output.stop()
        self._enabled = False

    def _download(self, sound_pack: str) -> None:
        sound_pack_dir = self._SOUND_PACKS_DIR / sound_pack
        sound_pack_dir.mkdir(parents=True)

        async def download_sound(sound: Sound, url: str, client: httpx.AsyncClient) -> None:
            response = await client.get(url)
            response.raise_for_status()

            extension = url.rsplit(".", 1)[-1]
            if not extension.isalnum():
                # we must have failed to detect a valid extension - mark it as unknown
                extension = "unknown"

            with (sound_pack_dir / f"{sound}.{extension}").open("wb") as file:
                file.write(response.content)

        async def download_all_sounds() -> None:
            async with httpx.AsyncClient() as client:
                await asyncio.gather(
                    *(download_sound(sound, url, client) for sound, url in self._ONLINE_SOUND_PACKS[sound_pack].items())
                )

        asyncio.run(download_all_sounds())

    def _play_sound(self, sound: Sound, *, loop: bool = False, volume: float = 1) -> None:
        if sound_file := self._sound_files[sound]:
            if loop:
                self._audio_output.play_on_loop(sound_file, volume=volume)
            else:
                self._audio_output.play_once(sound_file, volume=volume)

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(
            publisher, DropMergeRule | FillLines | RotateRule | MoveRule | LevelTracker | PowerupRule | Gravity
        ) and (self._game_indices is None or publisher.game_index in self._game_indices)

    @override
    def notify(self, message: NamedTuple) -> None:
        if not self._enabled:
            return

        match message:
            case GravityStartedMessage():
                self._play_sound(Sound.GRAVITY)
            case MergeMessage() | GravityFinishedMessage():
                self._play_sound(Sound.MERGE)
            case NewLevelMessage(level=level) if level > 0:
                self._play_sound(Sound.NEXT_LEVEL)
            case StartingLineFillMessage(filled_lines=filled_lines, is_line_clear=is_line_clear):
                self._play_line_fill_sound(filled_lines, is_line_clear=is_line_clear)
            case MoveMessage():
                self._play_sound(Sound.MOVE)
            case RotateMessage():
                self._play_sound(Sound.ROTATE)
            case PowerupTriggeredMessage():
                self._play_sound(Sound.POWERUP)
            case _:
                pass

    def _play_line_fill_sound(self, filled_lines: list[int], *, is_line_clear: bool) -> None:
        if not is_line_clear:
            self._play_sound(Sound.LINE_FILL)
            return

        num_lines_tetris_clear = 4
        if filled_lines == list(range(filled_lines[0], filled_lines[0] + num_lines_tetris_clear)) and (
            tetris_clear_sound_file := self._sound_files[Sound.TETRIS_CLEAR]
        ):
            self._audio_output.play_once(tetris_clear_sound_file)
        elif line_clear_sound_file := self._sound_files[Sound.LINE_CLEAR]:
            self._audio_output.play_once(line_clear_sound_file)

    @override
    def should_be_called_by(self, game_index: int) -> bool:
        # even non-sounded games should call this; the only sound being produced by this is the game over sound which
        # makes sense to play even for non-sounded games
        return True

    @override
    def on_all_games_over(self) -> None:
        self._audio_output.stop()
        self._music_playing = False

        if self._enabled:
            self._play_sound(Sound.GAME_OVER)

    @override
    def on_game_over(self, game_index: int) -> None:
        if self._enabled:
            self._play_sound(Sound.GAME_OVER)

    @override
    def on_game_start(self, game_index: int) -> None:
        if self._enabled and not self._music_playing:
            self._start_playing_music()

        self._music_playing = True

    def _start_playing_music(self) -> None:
        music_files = [
            music_file
            for sound in (Sound.MUSIC1, Sound.MUSIC2, Sound.MUSIC3)
            if (music_file := self._sound_files[sound]) is not None
        ]
        if not music_files:
            return

        self._audio_output.play_on_loop(random.choice(music_files))
