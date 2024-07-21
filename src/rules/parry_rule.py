from board_manipulations.board_manipulation import BoardManipulation
from board_manipulations.gravity import Gravity
from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.controller import Action


class ParryRule:
    def __init__(
        self, leeway_frames: int = 1, reward_board_manipulation: BoardManipulation = Gravity(per_col_probability=0.2)
    ) -> None:
        self._just_merged = False
        self._last_merge_frame: int | None = 0
        self._leeway_frames = leeway_frames
        self._reward_board_manipulation = reward_board_manipulation
        self._already_applied = False
        self._quick_merge = False

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        if self._already_applied:
            return

        if self._just_merged:
            # this check is cheese prevention: don't allow pressing quick drop and parry at the same time, triggering
            # the instant merge and parrying it at the same time
            if not self._quick_merge or action_counter.held_since(Action(down=True)) != 1:
                self._last_merge_frame = frame_counter
            self._just_merged = False

        if self._last_merge_within_leeway_frames(frame_counter) and self._parry_press_started_within_leeway_frames(
            action_counter
        ):
            self._reward_board_manipulation.manipulate(board)
            # TODO change the interface of the Board class such that private methods don't need to be called here
            # -> basically implement clear_lines_rule then this is obsolete
            full_line_positions = board._get_full_line_positions_ordered_top_to_bottom()
            board._clear_lines(full_line_positions)
            self._already_applied = True

    def _last_merge_within_leeway_frames(self, frame_counter: int) -> bool:
        return self._last_merge_frame is not None and frame_counter - self._last_merge_frame <= self._leeway_frames

    def _parry_press_started_within_leeway_frames(self, action_counter: ActionCounter) -> bool:
        return 0 < action_counter.held_since(Action(confirm=True)) <= self._leeway_frames + 1

    def custom_message(self, message: str) -> None:
        if message in ("slow merge", "quick merge"):
            self._just_merged = True
            self._already_applied = False
            self._quick_merge = message == "quick merge"
