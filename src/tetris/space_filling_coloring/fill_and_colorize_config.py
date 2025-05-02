import contextlib
import random
import shutil
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from pprint import pformat
from typing import Self

import numpy as np
from numpy.typing import NDArray

from tetris.space_filling_coloring import drawer
from tetris.space_filling_coloring.fill_and_colorize import (
    fill,
    fill_and_colorize_concurrently,
    fill_and_colorize_subsequently,
)
from tetris.space_filling_coloring.four_colorizer import FourColorizer
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller


@dataclass
class FillColorizeMode:
    pass


@dataclass
class ColorizeConcurrently(FillColorizeMode):
    minimum_separation_steps: int = 0
    allow_coloring_retry: bool = True


@dataclass
class ColorizeSequentially(FillColorizeMode):
    pass


@dataclass
class DontColorize(FillColorizeMode):
    pass


@dataclass
class FillColorizeConfig:
    size: tuple[int, int] | None = None
    holes: Sequence[tuple[tuple[int, int], tuple[int, int]]] = ()
    fill_and_colorize_rng_seed: int | None = None
    inverted: bool = False
    top_left_tendency: bool = True
    mode: FillColorizeMode = field(default_factory=ColorizeConcurrently)

    @classmethod
    def create_and_coerce_to_make_fillable(  # noqa: PLR0913
        cls,
        size: tuple[int, int] | None = None,
        holes: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (),
        fill_and_colorize_rng_seed: int | None = None,
        *,
        inverted: bool = False,
        top_left_tendency: bool = True,
        mode: FillColorizeMode | None = None,
    ) -> Self:
        config = cls(
            size=size,
            holes=holes,
            fill_and_colorize_rng_seed=fill_and_colorize_rng_seed,
            inverted=inverted,
            top_left_tendency=top_left_tendency,
            mode=mode or ColorizeConcurrently(),
        )
        assert config.size is not None

        while (
            not TetrominoSpaceFiller.space_can_be_filled(config._generate_array().astype(np.int32) - 1)
            and config.size[0] > 1
        ):
            # if the space is not fillable, try to reduce the size of the space until it is fillable
            config.size = (config.size[0] - 1, config.size[1])

        if not TetrominoSpaceFiller.space_can_be_filled(config._generate_array().astype(np.int32) - 1):
            msg = (
                'The configuration of space size and holes creates an unfillable space (at least one "island" with) '
                "to be filled that has a size not divisible by 4."
            )
            raise ValueError(msg)

        return config

    def __post_init__(self) -> None:
        self.holes = [self._normalize_hole(hole) for hole in self.holes]

        if self.size is None:
            terminal_width, terminal_height = shutil.get_terminal_size()
            self.size = terminal_height - 1, terminal_width // 2

    @staticmethod
    def _normalize_hole(hole: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
        (y1, x1), (y2, x2) = hole
        return (min(y1, y2), min(x1, x2)), (max(y1, y2), max(x1, x2))

    @property
    def num_holes(self) -> int:
        return len(self.holes)

    @classmethod
    def create_random(
        cls,
        rng_seed: int | None = None,
        max_holes: int = 10,
        size_limits: tuple[tuple[int, int], tuple[int, int]] = ((10, 80), (10, 150)),
    ) -> Self:
        main_rng = random.Random(rng_seed)

        fill_and_colorize_rng_seed = main_rng.randrange(2**32)
        num_holes = main_rng.randint(0, max_holes)

        inverted = num_holes > 1 and main_rng.randint(0, 1) == 1
        top_left_tendency = main_rng.randint(0, 1) == 1

        match main_rng.randint(0, 2):
            case 0:
                mode: FillColorizeMode = ColorizeConcurrently(
                    minimum_separation_steps=main_rng.randint(0, 10) * 10,
                    allow_coloring_retry=main_rng.randint(0, 1) == 1,
                )
            case 1:
                mode = ColorizeSequentially()
            case 2:
                mode = DontColorize()
            case _:
                msg = "This should never happen."
                raise AssertionError(msg)

        while True:
            size = (
                main_rng.randint(*size_limits[0]),
                main_rng.randint(*size_limits[1]),
            )

            holes: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for _ in range(num_holes):
                y1 = main_rng.randint(0, size[0] - 1)
                x1 = main_rng.randint(0, size[1] - 1)
                y2 = main_rng.randint(y1 + 1, size[0])
                x2 = main_rng.randint(x1 + 1, size[1])
                holes.append(((y1, x1), (y2, x2)))

            config = cls(
                fill_and_colorize_rng_seed=fill_and_colorize_rng_seed,
                size=size,
                holes=holes,
                inverted=inverted,
                top_left_tendency=top_left_tendency,
                mode=mode,
            )
            if TetrominoSpaceFiller.space_can_be_filled(config._generate_array().astype(np.int32) - 1):
                return config

    def _generate_array(self) -> NDArray[np.bool]:
        assert self.size is not None

        array = np.ones(self.size, dtype=bool)

        for (y1, x1), (y2, x2) in self.holes:
            array[y1:y2, x1:x2] = False

        if self.inverted:
            array = ~array

        return array

    def execute_and_check(
        self, *, draw: bool = True, error_report_file: Path | None = None
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]] | None:
        try:
            filled_space, colored_space = self.execute(draw=draw)

            TetrominoSpaceFiller.validate_filled_space(filled_space)
            FourColorizer.validate_colored_space(colored_space, filled_space)
        except Exception:  # noqa: BLE001
            error_description = f"{pformat(self)}\n\n{traceback.format_exc()}\n\n\n"

            if error_report_file is not None:
                error_report_file.parent.mkdir(parents=True, exist_ok=True)
                with error_report_file.open("a") as f:
                    f.write(error_description)

            print()

            with contextlib.suppress(UnboundLocalError):
                drawer.draw_full_array_raw(filled_space)

            print(error_description)
            return None
        else:
            return filled_space, colored_space

    def execute(self, *, draw: bool = True) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        match self.mode:
            case ColorizeConcurrently(minimum_separation_steps, allow_coloring_retry):
                for filled_space, colored_space in fill_and_colorize_concurrently(
                    self._generate_array(),
                    use_rng=self.fill_and_colorize_rng_seed is not None,
                    rng_seed=self.fill_and_colorize_rng_seed,
                    top_left_tendency=self.top_left_tendency,
                    minimum_separation_steps=minimum_separation_steps,
                    allow_coloring_retry=allow_coloring_retry,
                ):
                    if draw:
                        drawer.draw_array_fancy(np.where(colored_space > 0, colored_space, filled_space))

            case ColorizeSequentially():
                filled_space, colored_space = fill_and_colorize_subsequently(
                    self._generate_array(),
                    use_rng=self.fill_and_colorize_rng_seed is not None,
                    rng_seed=self.fill_and_colorize_rng_seed,
                    top_left_tendency=self.top_left_tendency,
                    space_updated_callback=(
                        lambda filled_space, colored_space: drawer.draw_array_fancy(
                            np.where(colored_space > 0, colored_space, filled_space)
                            if colored_space is not None
                            else filled_space
                        )
                    )
                    if draw
                    else None,
                )

            case DontColorize():
                filled_space = fill(
                    self._generate_array(),
                    use_rng=self.fill_and_colorize_rng_seed is not None,
                    rng_seed=self.fill_and_colorize_rng_seed,
                    top_left_tendency=self.top_left_tendency,
                    space_updated_callback=(lambda filled_space: drawer.draw_array_fancy(filled_space))
                    if draw
                    else None,
                )
                colored_space = np.zeros_like(filled_space, dtype=np.uint8)

            case _:
                msg = "This should never happen."
                raise AssertionError(msg)

        if draw:
            drawer.draw_array_fancy(np.where(colored_space > 0, colored_space, filled_space))

        return filled_space, colored_space


def fuzz_test_fill_and_colorize(
    min_size: tuple[int, int] = (10, 10),
    max_size: tuple[int, int] | None = None,
    max_holes: int = 10,
    *,
    draw: bool = True,
    error_report_file: Path | None = None,
) -> None:
    """Continuously run space filling and colorizing with random configurations.

    Note: The function runs indefinitely, generating and testing configurations until manually stopped.

    Args:
        min_size: The minimum size (height, width) for the generated configurations. Defaults to (10, 10).
        max_size: The maximum size (height, width) for the generated configurations.
            If None, the terminal size is used as the upper limit. Defaults to None.
        max_holes: The maximum number of holes allowed in the generated configurations. Defaults to 10.
        draw: Whether to visually draw the generated configurations in the terminal.
            If True, the program will also pause for a second after each test to allow the user to view the result.
            If False, the function will print the number of tests performed so far.
            Defaults to True.
        error_report_file: A file path to save error reports if a configuration fails.
            If None, no error reports are saved. Defaults to None.
    """
    min_height, min_width = min_size

    def _size_limits() -> tuple[tuple[int, int], tuple[int, int]]:
        """Determine size limits based on terminal size or default values."""
        if max_size:
            max_height, max_width = max_size
            return (min(min_height, max_height), max_height), (min(min_width, max_width), max_width)

        terminal_width, terminal_height = shutil.get_terminal_size()
        max_height, max_width = terminal_height - 1, terminal_width // 2
        return (min(min_height, max_height), max_height), (min(min_width, max_width), max_width)

    for i in count():
        FillColorizeConfig.create_random(size_limits=_size_limits(), max_holes=max_holes).execute_and_check(
            draw=draw, error_report_file=error_report_file
        )
        if draw:
            # if we draw, sleep to let the user look at the result drawn for a second
            time.sleep(1)
        else:
            # otherwise, print the number of tests performed so far
            print(f"\r{i}", end="")
