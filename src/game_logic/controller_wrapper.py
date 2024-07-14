
import numpy as np
from game_logic.interfaces.controller import Controller, Action

class ControllerWrapper:
    def __init__(self, controller: Controller) -> None:
        self._controller = controller
        self._actions_held_since = np.zeros(len(Action._fields), dtype=np.uint16)
    
    def update(self) -> None:
        self._actions_held_since = np.where(self._controller.get_action(), self._actions_held_since + 1, 0)
    
    def held_since(self, action: Action) -> int:
        return int(np.min(self._actions_held_since[np.where(action)]))
