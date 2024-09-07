import random

from tetris.game_logic.interfaces.controller import Action, Controller


class RandomController(Controller):
    MAX_VALID_COMBINATION_BUTTONS = 3  # at most 3 buttons can be pressed when only valid combinations are allowed
    NUM_BUTTONS = len(Action._fields)

    def __init__(
        self,
        p_do_nothing: float = 0,
        max_buttons_at_once: int | None = None,
        only_valid_combinations: bool = False,
    ) -> None:
        self._p_do_nothing = p_do_nothing
        self._only_valid_combinations = only_valid_combinations

        if max_buttons_at_once is not None and max_buttons_at_once > self.NUM_BUTTONS:
            raise ValueError(f"max_buttons_at_once cannot be greater than the number of buttons ({self.NUM_BUTTONS})")

        if (
            max_buttons_at_once is not None
            and only_valid_combinations
            and max_buttons_at_once > self.MAX_VALID_COMBINATION_BUTTONS
        ):
            raise ValueError(
                "max_buttons_at_once must cannot be greater than the biggest button combination that is valid "
                f"({self.MAX_VALID_COMBINATION_BUTTONS}), if only valid combinations are allowed",
            )

        self._max_buttons_at_once = max_buttons_at_once or self.MAX_VALID_COMBINATION_BUTTONS

    def get_action(self) -> Action:
        if random.random() < self._p_do_nothing:
            return Action()

        num_buttons = random.randint(1, self._max_buttons_at_once)
        while True:
            action = self._random_action(num_buttons)
            if not self._only_valid_combinations or self._is_valid(action):
                return action

    @staticmethod
    def _random_action(num_pressed_buttons: int) -> Action:
        pressed_button_indices = random.sample(range(RandomController.NUM_BUTTONS), num_pressed_buttons)
        return Action(*(i in pressed_button_indices for i in range(RandomController.NUM_BUTTONS)))

    @staticmethod
    def _is_valid(action: Action) -> bool:
        return not (action.left and action.right) and not (action.right_shoulder and action.left_shoulder)
