from itertools import cycle
from unittest.mock import Mock

import pytest

from game_logic.components.block import Block, BlockType
from game_logic.components.board import Board
from game_logic.game import Game, GameOver
from game_logic.interfaces.controller import Action, Controller
from game_logic.interfaces.rule_sequence import RuleSequence
from rules.move_rotate_rules import HeldInputPolicy, MoveRule, RotateRule
from rules.spawn_drop_merge_rule import SpawnDropMergeRule, SpawnStrategyImpl


class DummyController(Controller):
    def get_action(self) -> Action:
        return Action()


@pytest.fixture
def dummy_game() -> Game:
    return Game(
        ui=Mock(),
        board=Board.create_empty(20, 10),
        controller=DummyController(),
        clock=Mock(),
        rule_sequence=RuleSequence([]),
    )


def test_advance_frame_increases_fame_counter(dummy_game: Game) -> None:
    dummy_game.advance_frame(Action())
    assert dummy_game.frame_counter == 1

    dummy_game.advance_frame(Action())
    assert dummy_game.frame_counter == 2


def test_game_runs_as_expected() -> None:
    # GIVEN a 10x5 board
    board = Board.create_empty(10, 5)

    # GIVEN a predefined list of blocks to spawn
    blocks_to_spawn = cycle(
        [
            Block(BlockType.J),
            Block(BlockType.I),
            Block(BlockType.L),
            Block(BlockType.L),
            Block(BlockType.I),
            Block(BlockType.S),
            Block(BlockType.SQUARE),
            Block(BlockType.Z),
            Block(BlockType.I),
            Block(BlockType.T),
        ]
    )

    # GIVEN predefined controller actions
    actions = cycle(
        [
            Action(),
            Action(),
            Action(right=True),
            Action(right=True),
            Action(right_shoulder=True),
            Action(right_shoulder=True),
            Action(left=True),
            Action(left=True),
            Action(left_shoulder=True),
            Action(left_shoulder=True),
        ]
    )

    ui_mock = Mock()

    trigger_every_frame_policy = HeldInputPolicy(repeat_interval_frames=1)

    game = Game(
        ui=ui_mock,
        board=board,
        controller=DummyController(),
        clock=Mock(),
        rule_sequence=RuleSequence(
            [
                # ensure every single input is counted, even on adjacent frames
                MoveRule(held_input_policy=trigger_every_frame_policy),
                RotateRule(held_input_policy=trigger_every_frame_policy),
                SpawnDropMergeRule(
                    normal_interval=1,
                    spawn_strategy=SpawnStrategyImpl(select_block_fn=Mock(side_effect=blocks_to_spawn)),
                ),
            ]
        ),
    )

    expected_board_states = [
        # no action
        [
            ".XXX.",
            "...X.",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # no action
        [
            ".....",
            ".XXX.",
            "...X.",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # move right
        [
            ".....",
            ".....",
            "..XXX",
            "....X",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # move right
        [
            ".....",
            ".....",
            ".....",
            "..XXX",
            "....X",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # rotate right
        [
            ".....",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "..XX.",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # rotate right
        [
            ".....",
            ".....",
            ".....",
            ".....",
            "..X..",
            "..XXX",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
        # move left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".X...",
            ".XXX.",
            ".....",
            ".....",
            ".....",
        ],
        # move left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "X....",
            "XXX..",
            ".....",
            ".....",
        ],
        # rotate left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".X...",
            ".X...",
            "XX...",
        ],
        # rotate left (goes through before ground hit detection on the same frame)
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # no action
        [
            ".XXXX",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # no action
        [
            ".....",
            ".XXXX",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # move right
        [
            ".....",
            ".....",
            ".XXXX",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # move right
        [
            ".....",
            ".....",
            ".....",
            ".XXXX",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # rotate right
        [
            ".....",
            ".....",
            "..X..",
            "..X..",
            "..X..",
            "..X..",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # rotate right
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".XXXX",
            ".....",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # move left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXXX.",
            ".....",
            ".....",
            "XXX..",
            "..X..",
        ],
        # move left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "XXXX.",
            ".....",
            "XXX..",
            "..X..",
        ],
        # rotate left (would overlap with existing block, but gets nudged)
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate left (would overlap with existing block and cannot be nudged -> fails and block is merged)
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # no action
        [
            ".XXX.",
            ".X...",
            ".....",
            ".....",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # no action
        [
            ".....",
            ".XXX.",
            ".X...",
            ".....",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move right
        [
            ".....",
            ".....",
            "..XXX",
            "..X..",
            ".....",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move right
        [
            ".....",
            ".....",
            ".....",
            "..XXX",
            "..X..",
            ".....",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate right
        [
            ".....",
            ".....",
            ".....",
            "..XX.",
            "...X.",
            "...X.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate right
        [
            ".....",
            ".....",
            ".....",
            ".....",
            "....X",
            "..XXX",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move left
        [
            ".....",
            ".....",
            ".....",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move left (no effect, block is just being spawned)
        [
            ".XXX.",
            ".X...",
            ".....",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate left
        [
            "..X..",
            "..X..",
            "..XX.",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate left
        [
            ".....",
            "...X.",
            ".XXX.",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # no action
        [
            ".....",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # no action
        [
            ".....",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move right (no effect, block is just being spawned)
        [
            ".XXXX",
            ".....",
            "...X.",
            ".XXX.",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # move right
        [
            ".....",
            ".XXXX",
            "...X.",
            ".XXX.",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
        # rotate right
        [
            "..X..",
            "..X..",
            "..XX.",
            ".XXX.",
            "...X.",
            ".XXX.",
            "...X.",
            "...X.",
            "XXXX.",
            "..XX.",
        ],
    ]

    for idx, (action, expected_board_state) in enumerate(zip(actions, expected_board_states, strict=False), start=1):
        # WHEN advancing the game frame by frame
        game.advance_frame(action)
        print(f"Actual board after step {idx}:", str(board), sep="\n", end="\n\n")

        # THEN the board state is as expected on every frame
        assert str(board) == "\n".join(expected_board_state)
        # THEN the UI been told to draw the game at every frame
        assert ui_mock.draw.call_count == idx

    # THEN the game raises GameOver after the last game-ending board-state
    with pytest.raises(GameOver):
        game.advance_frame(Action())

    # THEN the frame counter matches the number of frames/board states
    assert game.frame_counter == len(expected_board_states)
