from typing import Iterable, Mapping
from unittest.mock import Mock

import pytest
from game_logic.controller_wrapper import ControllerWrapper
from game_logic.interfaces.controller import Action


@pytest.fixture
def fake_controller() -> Mock:
    # fmt: off
    return Mock(
        get_action=Mock(
            side_effect=[
                Action( True,  True,  True,  True,  True),
                Action(False, False, False, False, False),
                Action( True,  True, False, False, False),
                Action( True,  True,  True, False,  True),
                Action( True,  True, False,  True,  True),
                Action(False, False, False,  True,  True),
            ]
        )
    )
    # fmt: on


def test_controller_wrapper_single_action_held_since(fake_controller: Mock) -> None:
    controller_wrapper = ControllerWrapper(fake_controller)
    assert_single_actions_held_for(controller_wrapper, (0, 0, 0, 0, 0))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (1, 1, 1, 1, 1))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (0, 0, 0, 0, 0))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (1, 1, 0, 0, 0))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (2, 2, 1, 0, 1))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (3, 3, 0, 1, 2))

    controller_wrapper.update()
    assert_single_actions_held_for(controller_wrapper, (0, 0, 0, 2, 3))


def assert_single_actions_held_for(
    controller_wrapper: ControllerWrapper, lengths: tuple[int, int, int, int, int]
) -> None:
    # fmt: off
    assert controller_wrapper.held_since(Action( True, False, False, False, False)) == lengths[0]
    assert controller_wrapper.held_since(Action(False,  True, False, False, False)) == lengths[1]
    assert controller_wrapper.held_since(Action(False, False,  True, False, False)) == lengths[2]
    assert controller_wrapper.held_since(Action(False, False, False,  True, False)) == lengths[3]
    assert controller_wrapper.held_since(Action(False, False, False, False,  True)) == lengths[4]
    # fmt: on


def test_controller_wrapper_action_combo_held_since(fake_controller: Mock) -> None:
    # fmt: off
    action_all = Action( True,  True,  True,  True,  True)
    action_01  = Action( True,  True, False, False, False)
    action_012 = Action( True,  True,  True, False, False)
    action_014 = Action( True,  True, False, False,  True)
    action_23  = Action(False, False,  True,  True, False)
    # fmt: on

    controller_wrapper = ControllerWrapper(fake_controller)
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 0, action_012: 0, action_014: 0, action_23: 0}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 1, action_01: 1, action_012: 1, action_014: 1, action_23: 1}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 0, action_012: 0, action_014: 0, action_23: 0}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 1, action_012: 0, action_014: 0, action_23: 0}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 2, action_012: 1, action_014: 1, action_23: 0}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 3, action_012: 0, action_014: 2, action_23: 0}
    )

    controller_wrapper.update()
    assert_actions_held_for(
        controller_wrapper, {action_all: 0, action_01: 0, action_012: 0, action_014: 0, action_23: 0}
    )


def assert_actions_held_for(controller_wrapper: ControllerWrapper, actions_lengths: Mapping[Action, int]) -> None:
    for action, length in actions_lengths.items():
        assert controller_wrapper.held_since(action) == length
