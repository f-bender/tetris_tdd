from game_logic.action_counter import ActionCounter
from game_logic.components.board import Board
from game_logic.interfaces.callback_collection import CallbackCollection
from game_logic.interfaces.controller import Action


class HeldInputPolicy:
    def __init__(self, repeat_interval_frames: int, single_press_delay_frames: int = 0) -> None:
        """Policy whether to trigger an action based on how long an input has been held for.

        Args:
            repeat_interval_frames: While the input is held down, how many frames there should be between the action
                being triggered.
            single_press_delay_frames: After the move input is initially pressed down, how many frames to wait before
                the action is repeated with the repeat interval. This is to avoid unintended double-triggering of the
                move action. If this value is lower than `repeat_interval_frames`, then there is no special delay, and
                the delay between the first and second trigger is equal to `repeat_interval_frames`. This is the default
                behaviour.
        """
        self._repeat_interval_frames = repeat_interval_frames
        self._single_press_delay_frames = max(repeat_interval_frames, single_press_delay_frames)

    def should_trigger(self, held_since_frames: int) -> bool:
        return (
            # trigger once immediately after start of press
            held_since_frames == 1
            or (
                # or if the single press delay has passed,
                held_since_frames > self._single_press_delay_frames
                # and the number of frames since it has passed is divisible by the repeat interval
                and (held_since_frames - self._single_press_delay_frames - 1) % self._repeat_interval_frames == 0
            )
        )


class MoveRule:
    def __init__(
        self,
        held_input_policy: HeldInputPolicy = HeldInputPolicy(repeat_interval_frames=4, single_press_delay_frames=15),
    ) -> None:
        """Initialize the Move Rule.

        Args:
            held_input_policy: Policy when to trigger the move action based on how long the corresponding button has
                been held for. The default values of repeat_interval_frames=4 and single_press_delay_frames=15 are
                fine-tuned for 60 FPS gameplay.
        """
        self._held_input_policy = held_input_policy

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        if not board.has_active_block():
            return

        if self._held_input_policy.should_trigger(held_since_frames=action_counter.held_since(Action(left=True))):
            board.try_move_active_block_left()

        if self._held_input_policy.should_trigger(held_since_frames=action_counter.held_since(Action(right=True))):
            board.try_move_active_block_right()


class RotateRule:
    def __init__(
        self,
        held_input_policy: HeldInputPolicy = HeldInputPolicy(repeat_interval_frames=20),
    ) -> None:
        """Initialize the Rotate Rule.

        Args:
            held_input_policy: Policy when to trigger the move action based on how long the corresponding button has
                been held for. The default value of repeat_interval_frames=20 is fine-tuned for 60 FPS gameplay.
        """
        self._held_input_policy = held_input_policy

    def apply(
        self, frame_counter: int, action_counter: ActionCounter, board: Board, callback_collection: CallbackCollection
    ) -> None:
        if not board.has_active_block():
            return

        if self._held_input_policy.should_trigger(
            held_since_frames=action_counter.held_since(Action(left_shoulder=True))
        ):
            board.try_rotate_active_block_left()

        if self._held_input_policy.should_trigger(
            held_since_frames=max(
                action_counter.held_since(Action(right_shoulder=True)),
                action_counter.held_since(Action(up=True)),  # up can be used for right rotation as well
            )
        ):
            board.try_rotate_active_block_right()
