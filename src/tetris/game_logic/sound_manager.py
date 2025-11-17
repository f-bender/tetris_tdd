import asyncio
import random
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import httpx

from tetris.game_logic.interfaces.audio_output import AudioOutput
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines
from tetris.game_logic.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.game_logic.rules.messages import (
    MoveMessage,
    NewLevelMessage,
    RotateMessage,
    StartingLineClearMessage,
    StartMergeMessage,
)


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

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return isinstance(publisher, SpawnDropMergeRule | ClearFullLines | RotateRule | MoveRule | LevelTracker) and (
            self._game_indices is None or publisher.game_index in self._game_indices
        )

    def notify(self, message: NamedTuple) -> None:
        if not self._enabled:
            return

        match message:
            case StartMergeMessage():
                self._play_sound(Sound.MERGE)
            case NewLevelMessage(level=level) if level > 0:
                self._play_sound(Sound.NEXT_LEVEL)
            case StartingLineClearMessage(cleared_lines=cleared_lines):
                num_lines_tetris_clear = 4
                if len(cleared_lines) == num_lines_tetris_clear and (
                    tetris_clear_sound_file := self._sound_files[Sound.TETRIS_CLEAR]
                ):
                    self._audio_output.play_once(tetris_clear_sound_file)
                elif line_clear_sound_file := self._sound_files[Sound.LINE_CLEAR]:
                    self._audio_output.play_once(line_clear_sound_file)
            case MoveMessage():
                self._play_sound(Sound.MOVE)
            case RotateMessage():
                self._play_sound(Sound.ROTATE)
            case _:
                pass

    def should_be_called_by(self, game_index: int) -> bool:
        return self._game_indices is None or game_index in self._game_indices

    def on_game_over(self) -> None:
        self._audio_output.stop()
        self._music_playing = False

        if self._enabled:
            self._play_sound(Sound.GAME_OVER)

    def on_game_start(self) -> None:
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
