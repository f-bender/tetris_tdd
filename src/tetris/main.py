from tetris.clock.amortizing import AmortizingClock
from tetris.controllers.keyboard import KeyboardController
from tetris.controllers.llm.controller import LLMController
from tetris.controllers.llm.gemini import Gemini
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.logging import configure_logging
from tetris.rules.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.parry_rule import ParryRule
from tetris.rules.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.track_score_rule import TrackScoreRule
from tetris.ui.cli import CLI


def main() -> None:
    configure_logging()

    clock = AmortizingClock(fps=60, window_size=120)

    game1 = Game(
        CLI(),
        Board.create_empty(20, 10),
        # KeyboardController(
        #     action_to_keys={
        #         "left": ["a"],
        #         "right": ["d"],
        #         "up": ["w"],
        #         "down": ["s"],
        #         "left_shoulder": ["q"],
        #         "right_shoulder": ["e"],
        #         "confirm": ["space"],  # numpad 0
        #         "cancel": ["ctrl"],
        #     }
        # ),
        LLMController(Gemini()),
        clock,
        *get_rules_and_callbacks(),
    )
    game2 = Game(
        CLI(offset=(0, 27)),
        Board.create_empty(20, 10),
        KeyboardController(
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
        ),
        clock,
        *get_rules_and_callbacks(),
    )
    while True:
        clock.tick()
        try:
            game1.advance_frame(game1._controller.get_action(game1._board))
        except GameOverError:
            game1.reset()
            game1._callback_collection.on_game_start()

        try:
            game2.advance_frame(game2._controller.get_action(game2._board))
        except GameOverError:
            game2.reset()
            game2._callback_collection.on_game_start()


def get_rules_and_callbacks() -> tuple[RuleSequence, CallbackCollection]:
    spawn_drop_merge_rule = SpawnDropMergeRule()
    parry_rule = ParryRule(leeway_frames=1)
    track_score_rule = TrackScoreRule()
    rule_sequence = RuleSequence(
        (
            MoveRule(),
            RotateRule(),
            spawn_drop_merge_rule,
            parry_rule,
            ClearFullLinesRule(),
            track_score_rule,
        ),
    )
    callback_collection = CallbackCollection((spawn_drop_merge_rule, parry_rule, track_score_rule))
    return rule_sequence, callback_collection


if __name__ == "__main__":
    main()
