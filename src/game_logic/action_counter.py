import numpy as np

from game_logic.interfaces.controller import Action


class ActionCounter:
    def __init__(self) -> None:
        self._actions_held_since = np.zeros(len(Action._fields), dtype=np.uint16)

    def update(self, action: Action) -> None:
        self._actions_held_since = np.where(action, self._actions_held_since + 1, 0)

    def held_since(self, action: Action) -> int:
        return int(np.min(self._actions_held_since[np.where(action)]))
