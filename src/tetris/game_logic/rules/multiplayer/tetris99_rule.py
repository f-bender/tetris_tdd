import random
from typing import NamedTuple, override

import numpy as np

from tetris.game_logic.action_counter import ActionCounter
from tetris.game_logic.components.board import Board
from tetris.game_logic.game import GameOverError
from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.dependency_manager import DependencyManager
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.interfaces.rule import Rule
from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines
from tetris.game_logic.rules.messages import (
    BoardTranslationMessage,
    FinishedLineFillMessage,
    Tetris99FromPowerup,
    Tetris99Message,
)
from tetris.game_logic.rules.special.powerup_effect import Tetris99LinePlaceEffect


class PlaceLinesManipulation:
    def __init__(self, num_lines: int) -> None:
        self._num_lines = num_lines

    def manipulate(self, board: Board) -> None:
        board_array = board.array_view_without_active_block().copy()

        game_over = np.any(board_array[: self._num_lines])

        row_to_fill_in = np.ones_like(board_array[0]) * board.NEUTRAL_BLOCK_INDEX
        row_to_fill_in[random.randrange(len(board_array[0]))] = 0

        board_array[: -self._num_lines] = board_array[self._num_lines :]
        board_array[-self._num_lines :] = row_to_fill_in

        board.set_from_array(board_array, active_block_displacement=(-self._num_lines, 0))

        if game_over:
            raise GameOverError


class Tetris99Rule(Publisher, Subscriber, Callback, Rule):
    def __init__(
        self,
        target_idxs: list[int],
        targeted_by_idxs: list[int] | None = None,
        *,
        self_targeting_when_alone: bool = False,
    ) -> None:
        """Initialize a Tetris99Rule.

        Args:
            target_idxs: List of game indices that this game can target when sending lines.
            targeted_by_idxs: List of game indices that can target this game when sending lines.
                If None, defaults to target_idxs.
            self_targeting_when_alone: If True, this game will target itself when no other games are alive.
                This ensures that a multiplayer game can't continue indefinitely.
        """
        super().__init__()

        if not target_idxs:
            msg = "At least one target_id must be provided."
            raise ValueError(msg)

        self._target_idxs = target_idxs
        self._targeted_by_idxs = targeted_by_idxs or target_idxs
        self._alive_target_idxs: list[int] = []

        self._num_recently_cleared_lines = 0
        self._num_lines_to_place = 0

        self._self_targeting_when_alone = self_targeting_when_alone

    @override
    def should_be_called_by(self, game_index: int) -> bool:
        # get called by all games, to know which game is still alive, and which is game over
        return game_index != DependencyManager.RUNTIME_INDEX

    @override
    def on_game_start(self, game_index: int) -> None:
        if game_index == self.game_index:
            self._num_recently_cleared_lines = 0
            self._num_lines_to_place = 0
        elif game_index in self._target_idxs and game_index not in self._alive_target_idxs:
            self._alive_target_idxs.append(game_index)

    @override
    def on_game_over(self, game_index: int) -> None:
        if game_index in self._alive_target_idxs:
            self._alive_target_idxs.remove(game_index)

    @override
    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        return (
            (
                (isinstance(publisher, FillLines) and publisher.is_line_clearer)
                or isinstance(publisher, Tetris99LinePlaceEffect)
            )
            and publisher.game_index == self.game_index
        ) or (isinstance(publisher, Tetris99Rule) and publisher.game_index in self._targeted_by_idxs)

    @override
    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        num_clear_line_subscriptions = sum(1 for publisher in publishers if isinstance(publisher, FillLines))
        num_tetris_99_subscriptions = sum(1 for publisher in publishers if isinstance(publisher, Tetris99Rule))

        if num_clear_line_subscriptions != 1:
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {num_clear_line_subscriptions} "
                "FillLines (line clearer) subscriptions!"
            )
            raise RuntimeError(msg)

        if num_tetris_99_subscriptions != len(self._targeted_by_idxs):
            msg = (
                f"{type(self).__name__} of game {self.game_index} has {num_tetris_99_subscriptions} other Tetris99Rule "
                f"subscriptions, expected {len(self._targeted_by_idxs)}!"
            )
            raise RuntimeError(msg)

    @override
    def notify(self, message: NamedTuple) -> None:
        if isinstance(message, FinishedLineFillMessage):
            self._num_recently_cleared_lines += len(message.filled_lines)

        elif (isinstance(message, Tetris99Message) and message.target_id == self.game_index) or isinstance(
            message, Tetris99FromPowerup
        ):
            self._num_lines_to_place += message.num_lines

    @override
    def apply(self, frame_counter: int, action_counter: ActionCounter, board: Board) -> None:
        if self._num_recently_cleared_lines:
            if self._alive_target_idxs:
                self.notify_subscribers(
                    Tetris99Message(
                        num_lines=self._num_recently_cleared_lines,
                        target_id=random.choice(self._alive_target_idxs),
                    )
                )
            elif self._self_targeting_when_alone:
                self._num_lines_to_place += self._num_recently_cleared_lines

            self._num_recently_cleared_lines = 0

        if self._num_lines_to_place:
            PlaceLinesManipulation(self._num_lines_to_place).manipulate(board)
            self.notify_subscribers(BoardTranslationMessage(y_offset=-self._num_lines_to_place))
            self._num_lines_to_place = 0
