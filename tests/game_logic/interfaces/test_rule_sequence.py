from unittest.mock import Mock

from tetris.game_logic.interfaces.rule_sequence import RuleSequence


def test_rule_sequence_calls_apply_in_order() -> None:
    # given a rule sequence with three rules
    calls: list[str] = []
    mock_rule_1 = Mock(apply=Mock(side_effect=lambda *_args, **_kwargs: calls.append("rule_1")))
    mock_rule_2 = Mock(apply=Mock(side_effect=lambda *_args, **_kwargs: calls.append("rule_2")))
    mock_rule_3 = Mock(apply=Mock(side_effect=lambda *_args, **_kwargs: calls.append("rule_3")))
    mock_action_counter = Mock()
    mock_board = Mock()
    rule_sequence = RuleSequence([mock_rule_1, mock_rule_2, mock_rule_3])

    # WHEN applying the rule sequence
    rule_sequence.apply(42, mock_action_counter, mock_board)

    # THEN the rules are applied
    mock_rule_1.apply.assert_called_once_with(42, mock_action_counter, mock_board)
    mock_rule_2.apply.assert_called_once_with(42, mock_action_counter, mock_board)
    mock_rule_3.apply.assert_called_once_with(42, mock_action_counter, mock_board)

    # and they are applied in order
    assert calls == ["rule_1", "rule_2", "rule_3"]
