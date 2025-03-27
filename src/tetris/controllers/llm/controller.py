import logging
import re
from queue import Queue
from threading import Thread
from typing import Protocol

from tetris.clock.amortizing import AmortizingClock
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller

LOGGER = logging.getLogger(__name__)


class LLM(Protocol):
    @property
    def requests_per_minute_limit(self) -> float: ...
    def start_new_chat(self, system_prompt: str | None) -> None: ...
    def send_message(self, message: str) -> str: ...


class LLMController(Controller):
    MOVE_KEY = "move"
    ROTATE_KEY = "rotate"

    LEFT_KEY = "left"
    RIGHT_KEY = "right"

    SYSTEM_PROMPT = (
        # high level description
        "You are a Tetris AI. You will be given the current state of a Tetris board and must decide what "
        "move(s) to make next. The board will look something like this:\n\n"
        f"{
            Board.from_string_representation('''
            ..........
            ..........
            ..XXX.....
            ...X......
            ..........
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        ''')
        }\n\n"
        "'X' represents a block, '.' represents an empty space. "
        "Note that this representation also always includes the currently falling block which you have control over. "
        "You must infer yourself where on the board this block is, and provide commands for this block going forward. "
        # command description
        "Your response must start with commands in the form of '<action> <direction> <count>', "
        f"where <action> is one of '{MOVE_KEY}' (move) or '{ROTATE_KEY}' (rotate), "
        f"and <direction> is one of '{LEFT_KEY}' (left) or '{RIGHT_KEY}' (right). "
        f"For example, '{MOVE_KEY} {LEFT_KEY} 3' means 'move left 3 times'. "
        "You can provide multiple commands in a single response, separated by commas. "
        f"For example, '{ROTATE_KEY} {LEFT_KEY} 2, {MOVE_KEY} {RIGHT_KEY} 3' "
        "means 'rotate left 2 times, then move right 3 times'. "
        "Rotations happen in 90 degree increments, meaning that 2 rotations correspond to a 180 degree rotation. "
        # add reasoning the the response
        "After the commands, you should put a newline character and then provide reasoning for your decision "
        "in the form of one or two sentences explaining why you chose the commands you did. "
        # example for a valid response
        "This would be an example for a valid response:\n\n"
        f"{MOVE_KEY} {LEFT_KEY} 4, {ROTATE_KEY} {RIGHT_KEY} 1\n"
        "I move left 4 times to line the block up vertically with the gap at the bottom of the board, and then "
        "rotate right once to make the block have the correct orientation to fit in the gap. The block can then simply "
        "drop into place.\n\n"
        # "emotional manipulation", trying to motivate the AI to do well
        "You have to clear lines and get a score of at least 10, otherwise I will lose my job and be homeless! "
        "Don't let me down! "
    )
    COMMAND_PATTERN = re.compile(
        rf"(?P<action>(?:{MOVE_KEY}|{ROTATE_KEY})) (?P<direction>(?:{LEFT_KEY}|{RIGHT_KEY})) (?P<count>\d+)"
    )

    def __init__(self, llm: LLM) -> None:
        self._board: Board | None = None
        self._active_frame = True
        self._last_response: str | None = None

        self._action_queue: Queue[Action] = Queue()
        Thread(target=self._continuously_ask_llm, args=(llm,), daemon=True).start()

    @property
    def symbol(self) -> str:
        return "✨"

    def _continuously_ask_llm(self, llm: LLM) -> None:
        llm.start_new_chat(self.SYSTEM_PROMPT)
        clock = AmortizingClock(fps=llm.requests_per_minute_limit / 60, window_size=10)
        while True:
            clock.tick()
            if self._board is None:
                continue

            try:
                commands_str = llm.send_message(str(self._board))
            except Exception:
                LOGGER.exception("Encountered exception while using LLM, will keep trying:")
                continue

            self._last_response = commands_str

            for command in commands_str.split("\n", maxsplit=1)[0].split(","):
                for parsed_action in self.parse_command(command.strip()):
                    self._action_queue.put(parsed_action)

    @staticmethod
    def parse_command(command: str) -> list[Action]:
        match = LLMController.COMMAND_PATTERN.match(command)
        if not match:
            LOGGER.warning("Invalid command: %s", command)
            return []

        action = match.group("action")
        direction = match.group("direction")
        count = int(match.group("count"))

        if action == LLMController.MOVE_KEY:
            return [Action(left=True) if direction == LLMController.LEFT_KEY else Action(right=True)] * count
        if action == LLMController.ROTATE_KEY:
            return [
                Action(left_shoulder=True) if direction == LLMController.LEFT_KEY else Action(right_shoulder=True)
            ] * count

        msg = f"Invalid action: {action}"
        raise ValueError(msg)

    def get_action(self, board: Board | None = None) -> Action:
        self._board = board
        if self._last_response:
            print("LLM response:", self._last_response)  # noqa: T201

        # skip every other frame such that each action is interpreted as a separate button press, instead of a single
        # button hold
        if not self._active_frame or self._action_queue.empty():
            self._active_frame = True
            return Action()

        self._active_frame = False
        return self._action_queue.get()
