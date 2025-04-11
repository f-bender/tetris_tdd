import contextlib
import random
import shutil
import traceback
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from pprint import pformat
from time import sleep
from typing import Self

import numpy as np
from numpy.typing import NDArray

from tetris.space_filling_coloring import drawer
from tetris.space_filling_coloring.concurrent_fill_and_colorize import fill_and_colorize
from tetris.space_filling_coloring.four_colorizer import FourColorizer
from tetris.space_filling_coloring.tetromino_space_filler import TetrominoSpaceFiller


@dataclass
class TestConfig:
    fill_and_colorize_rng_seed: int
    minimum_separation_steps: int
    size: tuple[int, int]
    holes: list[tuple[int, int, int, int]]
    inverted: bool = False

    def __post_init__(self) -> None:
        self.holes = [self._normalize_hole(hole) for hole in self.holes]

    @staticmethod
    def _normalize_hole(hole: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        y1, x1, y2, x2 = hole
        return min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)

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
        minimum_separation_steps = main_rng.randint(0, 10) * 10

        while True:
            size = (
                main_rng.randint(*size_limits[0]),
                main_rng.randint(*size_limits[1]),
            )

            holes: list[tuple[int, int, int, int]] = []
            for _ in range(num_holes):
                y1 = main_rng.randint(0, size[0] - 1)
                x1 = main_rng.randint(0, size[1] - 1)
                y2 = main_rng.randint(y1 + 1, size[0])
                x2 = main_rng.randint(x1 + 1, size[1])
                holes.append((y1, x1, y2, x2))

            config = cls(
                fill_and_colorize_rng_seed=fill_and_colorize_rng_seed,
                minimum_separation_steps=minimum_separation_steps,
                size=size,
                holes=holes,
                inverted=inverted,
            )
            if TetrominoSpaceFiller.space_can_be_filled(config.generate_array().astype(np.int32) - 1):
                return config

    def generate_array(self) -> NDArray[np.bool]:
        array = np.ones(self.size, dtype=bool)

        for y1, x1, y2, x2 in self.holes:
            array[y1:y2, x1:x2] = False

        if self.inverted:
            array = ~array

        return array


def fuzz_test(
    *,
    size_limits: tuple[tuple[int, int], tuple[int, int]] | None = None,
    max_holes: int = 10,
    draw: bool = False,
) -> None:
    determine_size_limits_from_terminal = False
    if size_limits is None:
        if draw:
            determine_size_limits_from_terminal = True
        else:
            size_limits = (10, 80), (10, 150)

    for i in count(1):
        if determine_size_limits_from_terminal:
            terminal_width, terminal_height = shutil.get_terminal_size()
            max_height, max_width = terminal_height - 1, terminal_width // 2
            size_limits = (min(max_height, 10), max_height), (min(max_width, 10), max_width)

        assert size_limits is not None
        _seeded_test(test_config=TestConfig.create_random(max_holes=max_holes, size_limits=size_limits), draw=draw)

        if draw:
            # if we draw, sleep to let the user look at the result drawn for a second
            sleep(1)
        else:
            # otherwise, print the number of tests performed so far
            print(f"\r{i}", end="")


def _seeded_test(
    test_config: TestConfig,
    *,
    error_output_file: Path | None = Path(__file__).parent / "fuzz_test_errors.txt",
    draw: bool = False,
) -> None:
    try:
        for filled_space, colored_space in fill_and_colorize(
            test_config.generate_array(),
            use_rng=True,
            rng_seed=test_config.fill_and_colorize_rng_seed,
            minimum_separation_steps=test_config.minimum_separation_steps,
        ):
            if draw:
                drawer.draw_array_fancy(np.where(colored_space > 0, colored_space, filled_space))

        if draw:
            drawer.draw_array_fancy(np.where(colored_space > 0, colored_space, filled_space))

        TetrominoSpaceFiller.validate_filled_space(filled_space)
        FourColorizer.validate_colored_space(colored_space, filled_space)
    except Exception:  # noqa: BLE001
        error_description = f"{pformat(test_config)}\n\n{traceback.format_exc()}\n\n\n"

        if error_output_file is not None:
            with error_output_file.open("a") as f:
                f.write(error_description)

        print()

        with contextlib.suppress(UnboundLocalError):
            drawer.draw_full_array_raw(filled_space)

        print(error_description)


if __name__ == "__main__":
    fuzz_test(draw=True)
