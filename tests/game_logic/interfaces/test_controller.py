from game_logic.interfaces.controller import Action, Controller


class DummyController(Controller):
    def get_action(self) -> Action:
        return Action()


def test_button_description_default_implementation() -> None:
    controller = DummyController()
    assert controller.get_button_description(Action()) == ""
    assert controller.get_button_description(Action(rotate_left=True)) == "Rotate Left"
    assert controller.get_button_description(Action(move_left=True, rotate_right=True)) == "Move Left + Rotate Right"
    assert (
        controller.get_button_description(
            Action(move_left=True, move_right=True, rotate_left=True, rotate_right=True, quick_drop=True)
        )
        == "Move Left + Move Right + Rotate Left + Rotate Right + Quick Drop"
    )
