from time import sleep

from tetris.clock.amortizing import AmortizingClock
from tetris.controllers.keyboard import KeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.logging import configure_logging
from tetris.rules.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.hacky_pause_rule import PauseRule
from tetris.rules.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.parry_rule import ParryRule
from tetris.rules.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.track_score_rule import TrackScoreRule
from tetris.ui.cli import CLI


def main() -> None:
    configure_logging()

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
        ),
    )
    callback_collection = CallbackCollection((spawn_drop_merge_rule, parry_rule, track_score_rule))

    game = Game(ui, board, controller, clock, rule_sequence, callback_collection)
    while True:
        game.run()
        game.reset()
        sleep(2)


if __name__ == "__main__":
    main()
