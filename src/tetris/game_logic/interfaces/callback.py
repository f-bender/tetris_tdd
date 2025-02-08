from tetris.game_logic.interfaces.dependency_manager import DEPENDENCY_MANAGER


class Callback:
    def __init__(self) -> None:
        super().__init__()

        self.game_index = DEPENDENCY_MANAGER.current_game_index
        DEPENDENCY_MANAGER.all_callbacks.append(self)

    def on_runtime_start(self) -> None: ...
    def on_game_start(self) -> None: ...

    def on_frame_start(self) -> None: ...
    def on_action_counter_updated(self) -> None: ...
    def on_rules_applied(self) -> None: ...
    def on_frame_end(self) -> None: ...

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index
