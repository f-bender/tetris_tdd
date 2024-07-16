from time import sleep

from clock.amortizing import AmortizingClock
from controllers.keyboard import KeyboardController
from game_logic.components import Board
from game_logic.game import Game
from game_logic.interfaces.rule_sequence import RuleSequence
from rules.move_rotate_rules import MoveRule, RotateRule
from rules.spawn_drop_merge_rule import SpawnDropMergeRule
from ui.cli import CLI


def main() -> None:
    Game(
        ui=CLI(),
        board=Board.create_empty(20, 10),
        controller=KeyboardController(),
        clock=AmortizingClock(fps=60, window_size=120),
        rule_sequence=RuleSequence(
            [
                MoveRule(),
                RotateRule(),
                SpawnDropMergeRule(),
            ]
        ),
    ).run()


if __name__ == "__main__":
    while True:
        main()
        sleep(1)
