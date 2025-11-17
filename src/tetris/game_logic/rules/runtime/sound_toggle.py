from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.interfaces.controller import Action
from tetris.game_logic.interfaces.runtime_rule import RuntimeRule
from tetris.game_logic.sound_manager import SoundManager


class SoundToggleRule(RuntimeRule):
    _SOUND_TOGGLE_ACTION = Action(up=True, confirm=True)

    def __init__(self, sound_manager: SoundManager) -> None:
        super().__init__()

        self._sound_manager = sound_manager

    def apply(self, frame_counter: int, action_counter: ActionCounter) -> None:
        if action_counter.held_since(self._SOUND_TOGGLE_ACTION) == 1:
            if self._sound_manager.is_enabled:
                self._sound_manager.disable()
            else:
                self._sound_manager.enable()
