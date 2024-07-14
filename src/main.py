from time import sleep

from clock.amortizing import AmortizingClock
from controllers.keyboard import KeyboardController
from game_logic.components import Board
from game_logic.game import Game
from ui.cli import CLI


def main() -> None:
    Game(
        ui=CLI(),
        board=Board.create_empty(20, 10),
        controller=KeyboardController(),
        clock=AmortizingClock(fps=60, window_size=120),
        initial_frame_interval=25,
    ).run()


if __name__ == "__main__":
    while True:
        main()
        sleep(1)
