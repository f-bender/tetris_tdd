from time import sleep

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
from ui.cli import CLI


def main() -> None:
    ui = CLI()
    board = Board.create_empty(20, 10)
    controller = KeyboardController()
    clock = AmortizingClock(fps=60, window_size=120)
    parry_rule = ParryRule(leeway_frames=1)
    rule_sequence = RuleSequence(
        (MoveRule(), RotateRule(), SpawnDropMergeRule(), PauseRule(controller, clock), parry_rule, ClearFullLinesRule())
    )
    callback_collection = CallbackCollection([parry_rule])

    Game(ui, board, controller, clock, rule_sequence, callback_collection).run()


if __name__ == "__main__":
    while True:
        main()
        sleep(1)
