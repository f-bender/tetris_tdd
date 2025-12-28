from typing import NamedTuple

from tetris.game_logic.interfaces.callback import Callback
from tetris.game_logic.interfaces.pub_sub import Publisher, Subscriber
from tetris.game_logic.rules.core.scoring.level_rule import LevelTracker
from tetris.game_logic.rules.messages import FinishedLineFillMessage, NewLevelMessage, ScoreMessage


class ScoreTracker(Callback, Publisher, Subscriber):
    def __init__(self) -> None:
        super().__init__()

        self._score = 0
        self._session_high_score = 0

        self._current_level = 0

        self._other_scores: dict[int, int] = {}  # game_index -> score

    @property
    def score(self) -> int:
        return self._score

    @property
    def session_high_score(self) -> int:
        return self._session_high_score

    def should_be_subscribed_to(self, publisher: Publisher) -> bool:
        from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines

        return (
            publisher.game_index == self.game_index
            and (
                isinstance(publisher, LevelTracker) or (isinstance(publisher, FillLines) and publisher.is_line_clearer)
            )
        ) or (publisher.game_index != self.game_index and isinstance(publisher, ScoreTracker))

    def verify_subscriptions(self, publishers: list[Publisher]) -> None:
        from tetris.game_logic.rules.board_manipulations.fill_lines import FillLines

        if not {type(publisher) for publisher in publishers} >= {FillLines, LevelTracker}:
            msg = (
                f"{type(self).__name__} of game {self.game_index} has unexpected subscriptions: {publishers}\n"
                "Expected one FillLines (line clearer) and one LevelTracker."
            )
            raise RuntimeError(msg)

    def on_game_start(self, game_index: int) -> None:
        self._score = 0
        self._notify_subscribers_about_score()

    def notify(self, message: NamedTuple) -> None:
        match message:
            case FinishedLineFillMessage(filled_lines=cleared_lines):
                self._score += self.compute_points(len(cleared_lines), self._current_level)
                self._session_high_score = max(self._session_high_score, self._score)

                self._notify_subscribers_about_score()
            case NewLevelMessage(level=level):
                self._current_level = level
            case ScoreMessage(score=other_score, game_index=other_game_index):
                if self._other_scores.get(other_game_index) != other_score:
                    self._other_scores[other_game_index] = other_score
                    self._notify_subscribers_about_score()
            case _:
                pass

    def _notify_subscribers_about_score(self) -> None:
        self.notify_subscribers(
            ScoreMessage(
                score=self._score,
                rank=1 + sum(1 for other_score in self._other_scores.values() if other_score > self._score),
                session_high_score=self._session_high_score,
                game_index=self.game_index,
            )
        )

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
