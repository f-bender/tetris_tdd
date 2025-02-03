import logging
import random
from collections.abc import Callable, Iterable

from tetris import logging_config
from tetris.controllers.heuristic_bot.controller import HeuristicBotController
from tetris.controllers.heuristic_bot.heuristic import Heuristic, mutated_heuristic
from tetris.game_logic.components import Board
from tetris.game_logic.game import Game, GameOverError
from tetris.game_logic.interfaces import global_current_game_index
from tetris.game_logic.interfaces.callback import ALL_CALLBACKS
from tetris.game_logic.interfaces.callback_collection import CallbackCollection
from tetris.game_logic.interfaces.pub_sub import ALL_PUBLISHERS, ALL_SUBSCRIBERS, Publisher
from tetris.game_logic.interfaces.rule_sequence import RuleSequence
from tetris.game_logic.interfaces.ui import UI
from tetris.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from tetris.rules.core.clear_full_lines_rule import ClearFullLinesRule
from tetris.rules.core.move_rotate_rules import MoveRule, RotateRule
from tetris.rules.core.spawn_drop_merge.spawn import SpawnStrategyImpl
from tetris.rules.core.spawn_drop_merge.spawn_drop_merge_rule import SpawnDropMergeRule
from tetris.rules.core.spawn_drop_merge.speed import SpeedStrategyImpl
from tetris.rules.monitoring.track_score_rule import TrackScoreCallback
from tetris.ui.cli.ui import CLI

LOGGER = logging.getLogger(__name__)


def main() -> None:
    HeuristicGym(50, ui=CLI()).run()


# TODO make this flexible enough that my multiprocessing idea works with it as well
# or, you know, identify the multiple responsibilities and split them up
class HeuristicGym:
    def __init__(
        self,
        population_size: int = 100,
        *,
        ui: UI | None = None,
        max_evaluation_frames: int = 100_000,
        board_size: tuple[int, int] = (20, 10),
        mutator: Callable[[Heuristic], Heuristic] = mutated_heuristic,
    ) -> None:
        self._ui = ui
        if self._ui:
            self._ui.initialize(*board_size, num_boards=population_size)

        self._max_evaluation_frames = max_evaluation_frames
        self._mutator = mutator

        self._heuristic_bot_controllers: list[HeuristicBotController] = []
        self._games: list[Game] = []
        # come up with a better way than keeping track of these separately?
        self._score_trackers: list[TrackScoreCallback] = []
        self._spawn_strategies: list[SpawnStrategyImpl] = []

        for idx in range(population_size):
            global_current_game_index.current_game_index = idx

            board = Board.create_empty(*board_size)
            controller = HeuristicBotController(board, lightning_mode=True)
            track_score_callback = TrackScoreCallback()
            spawn_strategy = SpawnStrategyImpl()

            self._heuristic_bot_controllers.append(controller)
            self._score_trackers.append(track_score_callback)
            self._spawn_strategies.append(spawn_strategy)
            self._games.append(
                Game(
                    board=board,
                    controller=controller,
                    rule_sequence=RuleSequence(
                        (
                            MoveRule(),
                            RotateRule(),
                            SpawnDropMergeRule(
                                spawn_strategy=spawn_strategy,
                                speed_strategy=SpeedStrategyImpl(base_interval=1),
                                spawn_delay=0,
                            ),
                            ClearFullLinesRule(),
                        )
                    ),
                    callback_collection=CallbackCollection((track_score_callback,)),
                )
            )

        _wire_up_pubs_subs()
        _wire_up_callbacks(self._games)

    def run(self) -> None:
        GeneticAlgorithm(
            initial_population=[Heuristic()] + [mutated_heuristic(Heuristic()) for _ in range(len(self._games) - 1)],  # type: ignore[call-arg]
            population_evaluator=self._evaluate_heuristics,
            mutator=self._mutator,
            post_evaluation_callback=self._post_evaluation_callback,
        ).run()

    def _evaluate_heuristics(self, heuristics: list[Heuristic]) -> list[float]:  # noqa: C901
        for controller, heuristic in zip(self._heuristic_bot_controllers, heuristics, strict=True):
            controller.heuristic = heuristic

        for game in self._games:
            game.reset()

        seed = random.randrange(2**32)
        for spawn_strategy in self._spawn_strategies:
            spawn_strategy.select_block_fn = SpawnStrategyImpl.truly_random_select_fn(seed)

        # TODO: extract the following core into separate method

        num_alive_games = len(self._games)
        # TODO instead of counting number of frames, count and limit number of blocks spawned
        # (need to listen (subscribe) to the spawn rules for that)
        for i in range(self._max_evaluation_frames):
            if i % 1000 == 0:
                LOGGER.debug("Frame %d", i)

            for game in self._games:
                if not game.alive:
                    continue

                try:
                    game.advance_frame()
                except GameOverError:
                    num_alive_games -= 1

            if self._ui:
                self._ui.advance_startup()
                self._ui.draw(game.board.as_array() for game in self._games)

            if num_alive_games == 0:
                break

        return [tracker.score for tracker in self._score_trackers]

    def _post_evaluation_callback(self, population: list[Heuristic], fitnesses: list[float]) -> None:
        sorted_population, sorted_fitnesses = zip(
            *sorted(
                zip(population, fitnesses, strict=True),
                key=lambda x: x[1],
                reverse=True,
            ),
            strict=True,
        )

        LOGGER.info("Sorted fitnesses: %s", sorted_fitnesses)
        LOGGER.info("Best heuristic '%r' has fitness '%f'", sorted_population[0], sorted_fitnesses[0])

        LOGGER.debug("Top heuristics: %r", sorted_population[:10])


def _wire_up_pubs_subs() -> None:
    for subscriber in ALL_SUBSCRIBERS:
        subscriptions: list[Publisher] = []

        for publisher in ALL_PUBLISHERS:
            if subscriber.should_be_subscribed_to(publisher):
                publisher.add_subscriber(subscriber)
                subscriptions.append(publisher)

        subscriber.verify_subscriptions(subscriptions)


def _wire_up_callbacks(games: Iterable[Game]) -> None:
    for idx, game in enumerate(games):
        game.callback_collection = CallbackCollection(
            tuple(callback for callback in ALL_CALLBACKS if callback.should_be_called_by(idx))
        )


if __name__ == "__main__":
    logging_config.configure_logging()
    main()
