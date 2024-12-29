from tetris.clock.amortizing import AmortizingClock
from tetris.controllers.keyboard import KeyboardController
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.runtime import Runtime
from tetris.logging_config import configure_logging
from tetris.rules.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.parry_rule import ParryRule
from tetris.rules.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.track_score_rule import TrackScoreRule
from tetris.ui.cli import CLI


def main() -> None:
    configure_logging()

    runtime_rules: list[Rule] = []
    runtime_callbacks: list[Callback] = []

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
    rule_sequence, callback_collection, _runtime_rules, _runtime_callbacks = get_rules_and_callbacks("Game 1")
    runtime_rules.extend(_runtime_rules)
    runtime_callbacks.extend(runtime_callbacks)

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
    rule_sequence, callback_collection, _runtime_rules, _runtime_callbacks = get_rules_and_callbacks("Game 2")
    runtime_rules.extend(_runtime_rules)
    runtime_callbacks.extend(runtime_callbacks)

    game_2 = Game(board, controller, rule_sequence, callback_collection)

    ui = CLI()
    clock = AmortizingClock(fps=60, window_size=120)
    runtime = Runtime(
        ui,
        clock,
        [game_1, game_2],
        KeyboardController(),
        rule_sequence=RuleSequence(runtime_rules),
        callback_collection=CallbackCollection(runtime_callbacks),
    )

    while True:
        runtime.run()


def get_rules_and_callbacks(
    name: str | None = None,
) -> tuple[RuleSequence, CallbackCollection, list[Rule], list[Callback]]:
    """Get rules and callbacks relevant for one instance of a game.

    Returns:
        - a RuleSequence to be passed to the Game
        - a CallbackCollection to be passed to the Game
        - a list of Rules to be put into the RuleSequence for the overall Runtime
        - a list of Callbacks to be put into the CallbackCollection for the overall Runtime
    """
    spawn_drop_merge_rule = SpawnDropMergeRule()
    parry_rule = ParryRule(leeway_frames=1)
    track_score_rule = TrackScoreRule(header=name)
    rule_sequence = RuleSequence(
        (
            MoveRule(),
            RotateRule(),
            spawn_drop_merge_rule,
            parry_rule,
            ClearFullLinesRule(),
        ),
    )
    callback_collection = CallbackCollection((spawn_drop_merge_rule, parry_rule, track_score_rule))
    return rule_sequence, callback_collection, [track_score_rule], []


if __name__ == "__main__":
    main()
