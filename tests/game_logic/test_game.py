from itertools import cycle
from unittest.mock import Mock

from tetris.game_logic.components.block import Block, BlockType
from tetris.game_logic.components.board import Board
from tetris.game_logic.game import Game
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.rules.core.drop_merge.drop_merge_rule import DropMergeRule
from tetris.game_logic.rules.core.drop_merge.speed import SpeedStrategyImpl
from tetris.game_logic.rules.core.move_rotate_rules import HeldInputPolicy, MoveRule, RotateRule
from tetris.game_logic.rules.core.post_merge.post_merge_rule import PostMergeRule
from tetris.game_logic.rules.core.spawn.spawn import SpawnRule


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
            Block(BlockType.O),
            Block(BlockType.Z),
            Block(BlockType.I),
            Block(BlockType.T),
        ],
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
        ],
    )

    controller_mock = Mock(get_action=Mock(side_effect=actions))

    trigger_every_frame_policy = HeldInputPolicy(repeat_interval_frames=1)

    game = Game(
        board=board,
        controller=controller_mock,
        rule_sequence=RuleSequence(
            [
                # ensure every single input is counted, even on adjacent frames
                MoveRule(held_input_policy=trigger_every_frame_policy),
                RotateRule(held_input_policy=trigger_every_frame_policy),
                SpawnRule(select_block_fn=Mock(side_effect=blocks_to_spawn)),
                DropMergeRule(speed_strategy=SpeedStrategyImpl(base_interval=1)),
                PostMergeRule(effect_duration_frames=1),
            ],
        ),
    )

    # disable subscription verification which would fail but is irrelevant for this test
    game._ui_aggregator.verify_subscriptions = Mock()  # type: ignore[method-assign] # noqa: SLF001

    DEPENDENCY_MANAGER.wire_up()

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

    for idx, expected_board_state in enumerate(expected_board_states):
        # WHEN advancing the game frame by frame
        game.advance_frame()
        print(f"Actual board after step {idx}:", str(board), sep="\n", end="\n\n")  # noqa: T201

        # THEN the board state is as expected on every frame
        assert str(board) == "\n".join(expected_board_state)
        assert game.frame_counter == idx + 1

    # THEN the game is no longer alive after the last game-ending board-state
    game.advance_frame()
    assert not game.alive
