import cProfile
import random
import shutil
from time import sleep

import numpy as np
from ansi import color, cursor
from numpy.typing import NDArray

from ansi_extensions import color as colorx
from ansi_extensions import cursor as cursorx
from clock.amortizing import AmortizingClock
from controllers.keyboard import KeyboardController
from game_logic.components import Board
from game_logic.game import Game
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.rule_sequence import RuleSequence
from rules.clear_full_lines_rule import ClearFullLinesRule
from rules.hacky_pause_rule import PauseRule
from rules.move_rotate_rules import MoveRule, RotateRule
from rules.parry_rule import ParryRule
from rules.spawn_drop_merge_rule import SpawnDropMergeRule
from rules.track_score_rule import TrackScoreRule
from ui.cli import CLI
from ui.cli.tetromino_space_filler import TetrominoSpaceFiller


def main() -> None:
    ui = CLI()
    board = Board.create_empty(20, 10)
    controller = KeyboardController()
    clock = AmortizingClock(fps=60, window_size=120)
    spawn_drop_merge_rule = SpawnDropMergeRule()
    parry_rule = ParryRule(leeway_frames=1)
    track_score_rule = TrackScoreRule()
    rule_sequence = RuleSequence(
        (
            MoveRule(),
            RotateRule(),
            spawn_drop_merge_rule,
            PauseRule(controller, clock),
            parry_rule,
            ClearFullLinesRule(),
            track_score_rule,
        )
    )
    callback_collection = CallbackCollection((spawn_drop_merge_rule, parry_rule, track_score_rule))

    game = Game(ui, board, controller, clock, rule_sequence, callback_collection)
    while True:
        game.run()
        game.reset()
        sleep(2)


last_drawn: NDArray[np.int32] | None = None
i = 0


def draw_tetromino_space(space: NDArray[np.int32], force: bool = False) -> None:
    global last_drawn, i
    i += 1
    if i % 1 != 0 and not force:
        return

    rd = random.Random()
    if last_drawn is None:
        print(cursor.goto(1, 1), end="")
        for row in space:
            print(
                "".join(
                    rd.seed(int(val))  # type: ignore[func-returns-value]
                    or colorx.bg.rgb_truecolor(rd.randrange(50, 150), rd.randrange(50, 150), rd.randrange(50, 150))
                    + "  "
                    + color.fx.reset
                    if val > 0
                    else "  "
                    for val in row
                )
            )
    else:
        for y, x in np.argwhere(space != last_drawn):
            print(cursor.goto(y + 1, x * 2 + 1), end="")
            print(
                rd.seed(int(val))
                or colorx.bg.rgb_truecolor(rd.randrange(50, 150), rd.randrange(50, 150), rd.randrange(50, 150))
                + "  "
                + color.fx.reset
                if (val := space[y, x]) > 0
                else "  ",
                end="",
                flush=True,
            )
            print(cursor.goto(space.shape[0] + 1) + cursorx.erase_to_end(""), end="")

    last_drawn = space.copy()


if __name__ == "__main__":
    w, h = shutil.get_terminal_size()
    space = np.zeros((h // 4 * 4, w // 2), dtype=np.int32)

    filler = TetrominoSpaceFiller(
        space,
        use_rng=True,
        # top_left_tendency=True,
        # rng_seed=13,
        space_updated_callback=lambda: draw_tetromino_space(space),
    )
    filler.fill()
    # for _ in filler.ifill():
    #     draw_tetromino_space(space)
    # cProfile.run("for _ in filler.ifill(): pass", sort="tottime")
    # cProfile.run("filler.fill()", sort="tottime")
    draw_tetromino_space(space, force=True)
