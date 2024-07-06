from clock.amortizing import AmortizingClock
from controllers.random import RandomController
from game_logic.components import Board
from game_logic.game import Game
from ui.cli import CLI


def main() -> None:
    Game(
        ui=CLI(),
        board=Board.create_empty(20, 10),
        controller=RandomController(p_do_nothing=0.5, only_valid_combinations=True),
        clock=AmortizingClock(fps=60, window_size=60),
        initial_frame_interval=1,
    ).run()


if __name__ == "__main__":
    while True:
        main()
