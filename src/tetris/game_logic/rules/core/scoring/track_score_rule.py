from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.messages import FinishedLineClearMessage, NewLevelMessage, ScoreMessage


class ScoreTracker(Callback, Publisher, Subscriber):
    def __init__(self) -> None:
        super().__init__()

        self._score = 0
        self._session_high_score = 0

        self._current_level = 0

    @property
    def score(self) -> int:
        return self._score

    @property
    def session_high_score(self) -> int:
        return self._session_high_score

    def should_be_called_by(self, game_index: int) -> bool:
        return game_index == self.game_index  # own game: for on_game_start

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines

        return isinstance(publisher, ClearFullLines | LevelTracker) and publisher.game_index == self.game_index

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        from tetris.game_logic.rules.board_manipulations.clear_lines import ClearFullLines

        if {type(publisher) for publisher in publishers} != {ClearFullLines, LevelTracker}:
            msg = (
                f"{type(self).__name__} of game {self.game_index} has unexpected subscriptions: {publishers}\n"
                "Expected one ClearFullLines and one LevelTracker."
            )
            raise RuntimeError(msg)

    def on_game_start(self) -> None:
        self._score = 0
        self.notify_subscribers(ScoreMessage(score=self._score, session_high_score=self._session_high_score))

    def notify(self, message: NamedTuple) -> None:
        match message:
            case FinishedLineClearMessage(cleared_lines=cleared_lines):
                self._score += self.compute_points(len(cleared_lines), self._current_level)
                self._session_high_score = max(self._session_high_score, self._score)

                self.notify_subscribers(ScoreMessage(score=self._score, session_high_score=self._session_high_score))
            case NewLevelMessage(level=level):
                self._current_level = level
            case _:
                pass

    @staticmethod
    def compute_points(num_cleared_lines: int, level: int) -> int:
        """Compute points based on number of cleared lines and current level.

        Formula taken from https://tetris.fandom.com/wiki/Scoring.
        """
        base_points_by_lines = {1: 40, 2: 100, 3: 300, 4: 1200}
        if num_cleared_lines in base_points_by_lines:
            base_points = base_points_by_lines[num_cleared_lines]
        else:
            # more than 4 lines (e.g. through a powerup) -> double the points for each additional line
            base_points = base_points_by_lines[4] * 2 ** (num_cleared_lines - 4)

        return base_points * (level + 1)
