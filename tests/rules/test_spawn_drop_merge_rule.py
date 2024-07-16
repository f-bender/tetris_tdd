from unittest.mock import Mock

from rules.spawn_drop_merge_rule import SpawnDropMergeRule


def test_nothing() -> None:
    drop_rule = SpawnDropMergeRule(normal_interval=6, quick_interval_factor=2)

    # fmt: off
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 0, quick_drop_held_since= 0, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 1, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 2, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 3, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 4, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 5, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 6, quick_drop_held_since= 0, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 7, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 8, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter= 9, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=10, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=11, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=12, quick_drop_held_since= 0, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=13, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=14, quick_drop_held_since= 1, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=15, quick_drop_held_since= 2, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=16, quick_drop_held_since= 3, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=17, quick_drop_held_since= 4, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=18, quick_drop_held_since= 5, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=19, quick_drop_held_since= 6, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=20, quick_drop_held_since= 7, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=21, quick_drop_held_since= 8, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=22, quick_drop_held_since= 9, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=23, quick_drop_held_since=10, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=24, quick_drop_held_since=11, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=25, quick_drop_held_since=12, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=26, quick_drop_held_since=13, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=27, quick_drop_held_since=14, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=28, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=29, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=30, quick_drop_held_since= 0, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=31, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=32, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=33, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=34, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=35, quick_drop_held_since= 0, should_trigger=False)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=36, quick_drop_held_since= 0, should_trigger= True)
    assert_whether_drop_triggered_after(drop_rule, frame_counter=37, quick_drop_held_since= 0, should_trigger=False)
    # fmt: on


def assert_whether_drop_triggered_after(
    drop_rule: SpawnDropMergeRule, frame_counter: int, quick_drop_held_since: int, should_trigger: bool
) -> None:
    mock_board = Mock()
    drop_rule.apply(
        frame_counter=frame_counter,
        action_counter=Mock(held_since=Mock(return_value=quick_drop_held_since)),
        board=mock_board,
    )
    if should_trigger:
        mock_board.drop_active_block.assert_called()
    else:
        mock_board.drop_active_block.assert_not_called()
