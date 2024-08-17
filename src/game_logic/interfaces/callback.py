from typing import NamedTuple


class Callback:
    def custom_message(self, message: NamedTuple) -> None: ...

    def on_game_start(self) -> None: ...

    def on_frame_start(self) -> None: ...
    def on_action_counter_updated(self) -> None: ...
    def on_rules_applied(self) -> None: ...
    def on_frame_end(self) -> None: ...
