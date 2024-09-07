from tetris.game_logic.interfaces.controller import Action, Controller


class DummyController(Controller):
    def get_action(self) -> Action:
        return Action()


def test_button_description_default_implementation() -> None:
    controller = DummyController()
    assert controller.get_button_description(Action()) == ""
    assert controller.get_button_description(Action(left_shoulder=True)) == "Left Shoulder"
    assert controller.get_button_description(Action(left=True, right_shoulder=True)) == "Left + Right Shoulder"
    assert (
        controller.get_button_description(
            Action(
                left=True,
                right=True,
                up=True,
                down=True,
                left_shoulder=True,
                right_shoulder=True,
                confirm=True,
                cancel=True,
            )
        )
        == "Left + Right + Up + Down + Left Shoulder + Right Shoulder + Confirm + Cancel"
    )
