from tetris.controllers.llm.controller import LLMController
from tetris.game_logic.interfaces.controller import Action


def test_parse_command() -> None:
    """Test the _parse_command method of LLMController."""
    commands = LLMController.parse_command("ML3")
    assert commands == [Action(left=True)] * 3

    commands = LLMController.parse_command("RR2")
    assert commands == [Action(right_shoulder=True)] * 2

    commands = LLMController.parse_command("INVALID")
    assert commands == []
