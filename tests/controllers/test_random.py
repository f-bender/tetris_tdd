from controllers.random import RandomController
from game_logic.interfaces.controller import Action


def test_do_nothing_ever() -> None:
    random_controller = RandomController(p_do_nothing=1)
    assert random_controller.get_action() == Action()


def test_max_buttons() -> None:
    random_controller = RandomController(max_buttons_at_once=2)
    for _ in range(50):
        assert sum(random_controller.get_action()) <= 2


def test_no_invalid_combinations() -> None:
    random_controller = RandomController(only_valid_combinations=True)
    for _ in range(50):
        action = random_controller.get_action()
        assert not (action.left and action.right)
        assert not (action.right_shoulder and action.left_shoulder)
