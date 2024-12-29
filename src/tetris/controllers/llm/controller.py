import logging
import re
from collections import deque
from itertools import chain
from threading import Lock, Thread
from typing import Protocol

from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.controller import Action, Controller

LOGGER = logging.getLogger(__name__)


class LLM(Protocol):
    def start_new_chat(self, system_prompt: str | None) -> None: ...
    def send_message(self, message: str) -> str: ...


class LLMController(Controller):
    SYSTEM_PROMPT = (
        "You are a Tetris AI. You will be given the current state of a Tetris board and must decide what "
        f"move(s) to make next. The board will look something like this:\n\n{Board.from_string_representation('''
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
        ''')}\n\n 'X' represents a block, '.' represents an empty space. You must return movement commands in the form "
        "of <action><direction><count>, where <action> is one of 'M' (move) or 'R' (rotate), and <direction> is one of "
        "'L' (left) or 'R' (right). For example, 'ML3' means 'move left 3 times'. You can provide multiple commands in "
        "a single response, separated by commas. "
        "For example, 'RL2,MR3' means 'rotate left 2 times, then move right 3 times'. "
        "You have to clear lines and get a score of at least 10, otherwise I will lose my job and be homeless! "
        "Don't let me down! "
        "If you want to skip a turn, respond with 'SKIP'. "
        "Your response should not contain ANYTHING other than these commands, "
        "meaning it should not contain any spaces, newlines, or any other than the specified characters."
    )
    COMMAND_PATTERN = re.compile(r"(?P<action>[MR])(?P<direction>[LR])(?P<count>\d+)")

    def __init__(self, llm: LLM) -> None:
        self._board: Board | None = None
        self._active_frame = True
        self._last_response: str | None = None

        self._action_queue: deque[Action] = deque()
        self._action_queue_lock = Lock()

        Thread(target=self._continuously_ask_llm, args=(llm,), daemon=True).start()

    def _continuously_ask_llm(self, llm: LLM) -> None:
        llm.start_new_chat(self.SYSTEM_PROMPT)
        while True:
            if self._board is None:
                continue

            commands_str = llm.send_message(str(self._board))
            self._last_response = commands_str

            if commands_str == "SKIP":
                continue

            parsed_actions = list(
                chain.from_iterable(self.parse_command(command) for command in commands_str.split(","))
            )

            with self._action_queue_lock:
                self._action_queue.extend(parsed_actions)

    @staticmethod
    def parse_command(command: str) -> list[Action]:
        match = LLMController.COMMAND_PATTERN.match(command)
        if not match:
            LOGGER.warning("Invalid command: %s", command)
            return []

        action = match.group("action")
        direction = match.group("direction")
        count = int(match.group("count"))

        if action == "M":
            return [Action(left=True) if direction == "L" else Action(right=True)] * count
        if action == "R":
            return [Action(left_shoulder=True) if direction == "L" else Action(right_shoulder=True)] * count

        msg = f"Invalid action: {action}"
        raise ValueError(msg)

    def get_action(self, board: Board | None = None) -> Action:
        self._board = board
        if self._last_response:
            print("LLM response:", self._last_response)  # noqa: T201

        # skip every other frame such that each action is interpreted as a separate button press, instead of a single
        # button hold
        if not self._active_frame or not self._action_queue:
            self._active_frame = True
            return Action()

        self._active_frame = False
        with self._action_queue_lock:
            return self._action_queue.popleft()
