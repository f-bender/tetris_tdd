from unittest.mock import Mock

from tetris.game_logic.game import PLAYING_STATE
from tetris.rules.move_rotate_rules import HeldInputPolicy, MoveRule


def test_move_triggered_on_correct_frames() -> None:
    move_rule = MoveRule(HeldInputPolicy(single_press_delay_frames=15, repeat_interval_frames=4))

    # fmt: off
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 0, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 1, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 2, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 3, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 4, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 5, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 6, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 7, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 8, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames= 9, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=10, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=11, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=12, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=13, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=14, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=15, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=16, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=17, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=18, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=19, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=20, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=21, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=22, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=23, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=24, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=25, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=26, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=27, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=28, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=29, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=30, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=31, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=32, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=33, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=34, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=35, should_trigger=False)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=36, should_trigger= True)
    assert_whether_move_triggered_after(move_rule=move_rule, n_frames=37, should_trigger=False)
    # fmt: on


def assert_whether_move_triggered_after(move_rule: MoveRule, *, n_frames: int, should_trigger: bool) -> None:
    mock_board = Mock()
    move_rule.apply(
        frame_counter=42,  # ignored by the move rule
        action_counter=Mock(held_since=Mock(return_value=n_frames)),
        board=mock_board,
        callback_collection=Mock(),
        state=PLAYING_STATE,
    )
    if should_trigger:
        mock_board.try_move_active_block_left.assert_called()
        mock_board.try_move_active_block_right.assert_called()
    else:
        mock_board.try_move_active_block_left.assert_not_called()
        mock_board.try_move_active_block_right.assert_not_called()
