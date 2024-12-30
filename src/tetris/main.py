from tetris.clock.amortizing import AmortizingClock
from tetris.controllers.keyboard import KeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.runtime import Runtime
from tetris.logging_config import configure_logging
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.monitoring.track_performance_rule import TrackPerformanceCallback
from tetris.rules.monitoring.track_score_rule import TrackScoreRule
from tetris.rules.multiplayer.tetris99_rule import Tetris99Rule
from tetris.rules.special.parry_rule import ParryRule
from tetris.ui.cli import CLI

FPS = 60


def main() -> None:
    configure_logging()

    t99_1 = Tetris99Rule(id=1, target_ids=[2])
    t99_2 = Tetris99Rule(id=2, target_ids=[1])

    t99_1.add_subscriber(t99_2)
    t99_2.add_subscriber(t99_1)

    board = Board.create_empty(20, 10)
    controller = KeyboardController(
        action_to_keys={
            "left": ["a"],
            "right": ["d"],
            "up": ["w"],
            "down": ["s"],
            "left_shoulder": ["q"],
            "right_shoulder": ["e"],
            "confirm": ["space"],
            "cancel": ["ctrl"],
        }
    )
    rule_sequence, callback_collection = get_rules_and_callbacks(t99_1, "Game 1")

    game_1 = Game(board, controller, rule_sequence, callback_collection)

    board = Board.create_empty(20, 10)
    controller = KeyboardController(
        action_to_keys={
            "left": ["left"],
            "right": ["right"],
            "up": ["up"],
            "down": ["down"],
            "left_shoulder": [],
            "right_shoulder": [],
            "confirm": ["enter", 82],  # numpad 0
            "cancel": ["esc"],
        }
    )
    rule_sequence, callback_collection = get_rules_and_callbacks(t99_2, "Game 2")

    game_2 = Game(board, controller, rule_sequence, callback_collection)

    ui = CLI()
    clock = AmortizingClock(fps=FPS, window_size=120)
    runtime = Runtime(
        ui,
        clock,
        [game_1, game_2],
        KeyboardController(),
        callback_collection=CallbackCollection((TrackPerformanceCallback(fps=FPS),)),
    )

    runtime.run()


def get_rules_and_callbacks(
    tetris99_rule: Tetris99Rule,
    name: str | None = None,
) -> tuple[RuleSequence, CallbackCollection]:
    """Get rules and callbacks relevant for one instance of a game.

    Returns:
        - a RuleSequence to be passed to the Game
        - a CallbackCollection to be passed to the Game
    """
    spawn_drop_merge_rule = SpawnDropMergeRule()
    parry_rule = ParryRule(leeway_frames=1)
    track_score_rule = TrackScoreRule(header=name)
    clear_full_lines_rule = ClearFullLinesRule()

    clear_full_lines_rule.add_subscriber(tetris99_rule)
    clear_full_lines_rule.add_subscriber(track_score_rule)
    spawn_drop_merge_rule.add_subscriber(parry_rule)

    rule_sequence = RuleSequence(
        (
            MoveRule(),
            RotateRule(),
            spawn_drop_merge_rule,
            parry_rule,
            clear_full_lines_rule,
            track_score_rule,
            tetris99_rule,
        ),
    )
    callback_collection = CallbackCollection((spawn_drop_merge_rule, track_score_rule))
    return rule_sequence, callback_collection


if __name__ == "__main__":
    main()
