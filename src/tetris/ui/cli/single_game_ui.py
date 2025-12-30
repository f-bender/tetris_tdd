from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from math import ceil
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.animations import (
    AnimationSpec,
    BlooperAnimationSpec,
    PowerupTriggeredAnimationSpec,
    TetrisAnimationSpec,
)
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.game_logic.rules.special.powerup import PowerupRule
from tetris.ui.cli.animations import (
    BlooperAnimation,
    PowerupTriggeredAnimation,
    TetrisAnimationLeft,
    TetrisAnimationRight,
)
from tetris.ui.cli.color_palette import ColorPalette
from tetris.ui.cli.vec import Vec


class Alignment(Enum):
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()


@dataclass(slots=True)
class Text:
    text: str
    position: Vec
    alignment: Alignment = Alignment.LEFT


@dataclass(slots=True)
class Overlay:
    position: Vec
    frame: NDArray[np.uint8]
    texts: list[Text] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]


class _Blink(NamedTuple):
    off_frames: int
    on_frames: int
    n_times: int


def generate_blink_off_frames(
    blinks: Sequence[_Blink] = (_Blink(10, 30, n_times=3), _Blink(10, 10, n_times=6), _Blink(5, 5, n_times=12)),
) -> frozenset[int]:
    result: set[int] = set()

    current_ttl = 1
    for blink in reversed(blinks):  # we start at ttl 1 (i.e. last frame) and go backward from there -> last blink first
        for _ in range(blink.n_times):
            current_ttl += blink.on_frames
            result.update(range(current_ttl, current_ttl + blink.off_frames))
            current_ttl += blink.off_frames

    return frozenset(result)


@dataclass(frozen=True)
class SingleGameUI:
    board_height: int
    board_width: int

    pixel_width: int
    total_num_games: int

    _BOARD_BORDER_WIDTH = 1

    _RIGHT_GAP_WIDTH = 2
    _RIGHT_ELEMENTS_WIDTH = 6

    _LINES_HEIGHT = 3
    _CONTROLLER_SYMBOL_HEIGHT = 3
    _DISPLAY_GAP_HEIGHT = 1

    @cached_property
    def _score_height(self) -> int:
        return 6 if self.total_num_games == 1 else 8

    _NEXT_BLOCK_HEIGHT = 5
    _LEVEL_HEIGHT = 4

    @cached_property
    def _game_over_overlay_height(self) -> int:
        return 7 if self.total_num_games == 1 else 8

    _GAME_OVER_OVERLAY_WIDTH = 12

    _POWERUP_TTL_VALUES_BLINKED_OFF = generate_blink_off_frames()

    # What the UI looks like:

    #                               board_width        RIGHT_ELEMENTS_WIDTH
    #                           <------------------>     <---------->
    #
    #                       ^  ______________________....____________  ^
    #          LINES_HEIGHT |  __Lines___________99__....____WASD____  | CONTROLLER_SYMBOL_HEIGHT
    #                       v  ______________________....____________  v
    #                          ......................................    | DISPLAY_GAP_HEIGHT
    #  BOARD_BORDER_WIDTH |    XXXXXXXXXXXXXXXXXXXXXX....____________  ^
    #                       ^  X####################X....__Top_______  |
    #                       |  X####################X....__99999999__  | SCORE_HEIGHT
    #                       |  X####################X....__Score_____  |
    #                       |  X####################X...._____12345__  |
    #                       |  X####################X....__Rank______  | [Rank: only if total_num_games > 1]
    #                       |  X####################X...._______2/5__  |
    #                       |  X####################X....____________  v
    #                       |  X####################X................    | DISPLAY_GAP_HEIGHT
    #          board_height |  X####################X....____________  ^
    #                       |  X####################X....__Next______  |
    #                       |  X####################X....____##______  | NEXT_BLOCK_HEIGHT
    #                       |  X####################X....__######____  |
    #                       |  X####################X....____________  v
    #                       |  X####################X................    | DISPLAY_GAP_HEIGHT
    #                       |  X####################X....____________  ^
    #                       |  X####################X....__Level_____  | LEVEL_HEIGHT
    #                       V  X####################X...._________9__  |
    #                          XXXXXXXXXXXXXXXXXXXXXX....____________  v
    #
    #                          -                     <-->
    #                  BOARD_BORDER_WIDTH       RIGHT_GAP_WIDTH

    # legend:
    # # board/block
    # X: board border
    # _ display background (black)
    # . gap (mask=False, filled by outer background)
    # note: 2 characters constitute one pixel

    @cached_property
    def board_background(self) -> NDArray[np.uint8]:
        board_background = np.full(
            (self.board_height, self.board_width), ColorPalette.index_of_color("board_bg_1"), dtype=np.uint8
        )

        board_background[1::2, ::2] = ColorPalette.index_of_color("board_bg_2")
        board_background[::2, 1::2] = ColorPalette.index_of_color("board_bg_2")

        return board_background

    @cached_property
    def total_size(self) -> tuple[int, int]:
        return (
            max(
                (
                    self._LINES_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._BOARD_BORDER_WIDTH
                    + self.board_height
                    + self._BOARD_BORDER_WIDTH
                ),
                (
                    self._CONTROLLER_SYMBOL_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._score_height
                    + self._DISPLAY_GAP_HEIGHT
                    + self._NEXT_BLOCK_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._LEVEL_HEIGHT
                ),
            ),
            self._BOARD_BORDER_WIDTH
            + self.board_width
            + self._BOARD_BORDER_WIDTH
            + self._RIGHT_GAP_WIDTH
            + self._RIGHT_ELEMENTS_WIDTH,
        )

    @cached_property
    def lines_position(self) -> Vec:
        return Vec(0, 0)

    @cached_property
    def board_border_position(self) -> Vec:
        return Vec(self._LINES_HEIGHT + self._DISPLAY_GAP_HEIGHT, 0)

    @cached_property
    def board_position(self) -> Vec:
        return self.board_border_position + Vec(self._BOARD_BORDER_WIDTH, self._BOARD_BORDER_WIDTH)

    @cached_property
    def board_with_border_height(self) -> int:
        return self._BOARD_BORDER_WIDTH + self.board_height + self._BOARD_BORDER_WIDTH

    @cached_property
    def board_with_border_width(self) -> int:
        return self._BOARD_BORDER_WIDTH + self.board_width + self._BOARD_BORDER_WIDTH

    @cached_property
    def controller_symbol_position(self) -> Vec:
        return Vec(0, self.board_with_border_width + self._RIGHT_GAP_WIDTH)

    @cached_property
    def score_position(self) -> Vec:
        return self.controller_symbol_position + Vec(self._CONTROLLER_SYMBOL_HEIGHT + self._DISPLAY_GAP_HEIGHT, 0)

    @cached_property
    def next_block_position(self) -> Vec:
        return self.score_position + Vec(self._score_height + self._DISPLAY_GAP_HEIGHT, 0)

    @cached_property
    def level_position(self) -> Vec:
        return self.next_block_position + Vec(self._NEXT_BLOCK_HEIGHT + self._DISPLAY_GAP_HEIGHT, 0)

    @cached_property
    def game_over_overlay_position(self) -> Vec:
        total_height, total_width = self.total_size
        return Vec(
            (total_height - self._game_over_overlay_height) // 2,
            (total_width - self._GAME_OVER_OVERLAY_WIDTH) // 2,
        )

    @cached_property
    def game_over_overlay_frame(self) -> NDArray[np.uint8]:
        game_over_frame = np.full(
            (self._game_over_overlay_height, self._GAME_OVER_OVERLAY_WIDTH),
            fill_value=ColorPalette.index_of_color("overlay_display_bg"),
            dtype=np.uint8,
        )
        game_over_frame[0, :] = game_over_frame[-1, :] = game_over_frame[:, 0] = game_over_frame[:, -1] = (
            ColorPalette.index_of_color("overlay_display_border")
        )
        return game_over_frame

    @cached_property
    def mask(self) -> NDArray[np.bool]:
        mask = np.zeros(self.total_size, dtype=np.bool)

        mask[
            self.lines_position.y : self.lines_position.y + self._LINES_HEIGHT,
            self.lines_position.x : self.lines_position.x + self.board_with_border_width,
        ] = True
        mask[
            self.board_border_position.y : self.board_border_position.y + self.board_with_border_height,
            self.board_border_position.x : self.board_border_position.x + self.board_with_border_width,
        ] = True
        mask[
            self.controller_symbol_position.y : self.controller_symbol_position.y + self._CONTROLLER_SYMBOL_HEIGHT,
            self.controller_symbol_position.x : self.controller_symbol_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.score_position.y : self.score_position.y + self._score_height,
            self.score_position.x : self.score_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.next_block_position.y : self.next_block_position.y + self._NEXT_BLOCK_HEIGHT,
            self.next_block_position.x : self.next_block_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.level_position.y : self.level_position.y + self._LEVEL_HEIGHT,
            self.level_position.x : self.level_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = True

        return mask

    def create_array_texts_animations(
        self, elements: SingleUiElements
    ) -> tuple[NDArray[np.uint8], list[Text], list[Overlay]]:
        ui_array = np.empty(self.total_size, dtype=np.uint8)
        texts: list[Text] = []

        self._add_lines_display(num_cleared_lines=elements.num_cleared_lines, ui_array=ui_array, texts=texts)
        self._add_board(board=elements.board, powerup_ttls=elements.powerup_ttls, ui_array=ui_array, texts=texts)
        self._add_board_border(ui_array)
        self._add_controller_symbol_display(
            controller_symbol=elements.controller_symbol, ui_array=ui_array, texts=texts
        )
        self._add_score_display(
            score=elements.score,
            session_high_score=elements.session_high_score,
            rank=elements.rank,
            ui_array=ui_array,
            texts=texts,
        )
        self._add_next_block_display(next_block=elements.next_block, ui_array=ui_array, texts=texts)
        self._add_level_display(level=elements.level, ui_array=ui_array, texts=texts)

        overlays = self._animation_overlays(elements.animations)
        if elements.game_over:
            overlays.append(self._game_over_overlay(score=elements.score, rank=elements.rank))

        return ui_array, texts, overlays

    def _add_lines_display(self, num_cleared_lines: int, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.lines_position.y : self.lines_position.y + self._LINES_HEIGHT,
            self.lines_position.x : self.lines_position.x + self.board_with_border_width,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Lines", position=self.lines_position + Vec(1, 1)))
        texts.append(
            Text(
                text=str(num_cleared_lines),
                position=self.lines_position + Vec(1, self.board_with_border_width - 1),
                alignment=Alignment.RIGHT,
            )
        )

    def _add_board(
        self, board: NDArray[np.uint8], powerup_ttls: dict[int, int], ui_array: NDArray[np.uint8], texts: list[Text]
    ) -> None:
        board_in_ui_array = np.where(board, board + ColorPalette.block_color_index_offset() - 1, self.board_background)

        # handle ghost block (indication where block would drop to)
        if np.any(np.isin(board, (Board.GHOST_BLOCK_CELL_VALUE, Board.POWERUP_GHOST_BLOCK_CELL_VALUE))):
            powerup_ghost = board == Board.POWERUP_GHOST_BLOCK_CELL_VALUE
            normal_ghost = board == Board.GHOST_BLOCK_CELL_VALUE

            # use ghost background color for ghost cells
            board_in_ui_array = np.where(
                powerup_ghost | normal_ghost,
                self.board_background
                + (ColorPalette.board_bg_ghost_index_offset() - ColorPalette.board_bg_index_offset()),
                board_in_ui_array,
            )

            for y, x in zip(*np.where(powerup_ghost), strict=True):
                texts.append(Text(text="?" * self.pixel_width, position=self.board_position + Vec(y, x)))

        # handle powerup blocks (custom dynamic color, and with "?" text on top)
        powerup_positions = np.where((board >= Board.MIN_POWERUP_CELL_VALUE) & (board <= Board.MAX_POWERUP_CELL_VALUE))
        for y, x in zip(*powerup_positions, strict=True):
            if powerup_ttls[int(board[y, x])] not in self._POWERUP_TTL_VALUES_BLINKED_OFF:
                board_in_ui_array[y, x] = ColorPalette.DYNAMIC_POWERUP_INDEX
                texts.append(Text(text="?" * self.pixel_width, position=self.board_position + Vec(y, x)))
            else:
                board_in_ui_array[y, x] = (
                    (board[y, x] % PowerupRule.POWERUP_SLOT_OFFSET) + ColorPalette.block_color_index_offset() - 1
                )

        # handle neutral blocks (used in Tetris99 to fill in garbage lines)
        board_in_ui_array[board == Board.NEUTRAL_BLOCK_INDEX] = ColorPalette.index_of_color("block_neutral")

        ui_array[
            self.board_position.y : self.board_position.y + self.board_height,
            self.board_position.x : self.board_position.x + self.board_width,
        ] = board_in_ui_array

    def _add_board_border(self, ui_array: NDArray[np.uint8]) -> None:
        ui_array[
            # top border
            self.board_border_position.y : self.board_border_position.y + self._BOARD_BORDER_WIDTH,
            self.board_border_position.x : self.board_border_position.x + self.board_with_border_width,
        ] = ui_array[
            # left border
            self.board_border_position.y : self.board_border_position.y + self.board_with_border_height,
            self.board_border_position.x : self.board_border_position.x + self._BOARD_BORDER_WIDTH,
        ] = ui_array[
            # bottom border
            self.board_border_position.y + self._BOARD_BORDER_WIDTH + self.board_height : self.board_border_position.y
            + self.board_with_border_height,
            self.board_border_position.x : self.board_border_position.x + self.board_with_border_width,
        ] = ui_array[
            # right border
            self.board_border_position.y : self.board_border_position.y + self.board_with_border_height,
            self.board_border_position.x + self._BOARD_BORDER_WIDTH + self.board_width : self.board_border_position.x
            + self.board_with_border_width,
        ] = ColorPalette.index_of_color("board_border")

    def _add_controller_symbol_display(
        self, controller_symbol: str, ui_array: NDArray[np.uint8], texts: list[Text]
    ) -> None:
        ui_array[
            self.controller_symbol_position.y : self.controller_symbol_position.y + self._CONTROLLER_SYMBOL_HEIGHT,
            self.controller_symbol_position.x : self.controller_symbol_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(
            Text(
                text=controller_symbol,
                position=self.controller_symbol_position + Vec(1, self._RIGHT_ELEMENTS_WIDTH // 2),
                alignment=Alignment.CENTER,
            )
        )

    def _add_score_display(
        self, score: int, session_high_score: int, rank: int, ui_array: NDArray[np.uint8], texts: list[Text]
    ) -> None:
        ui_array[
            self.score_position.y : self.score_position.y + self._score_height,
            self.score_position.x : self.score_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Top", position=self.score_position + Vec(1, 1)))
        texts.append(
            Text(
                text=str(session_high_score),
                position=self.score_position + Vec(2, self._RIGHT_ELEMENTS_WIDTH - 1),
                alignment=Alignment.RIGHT,
            )
        )
        texts.append(Text(text="Score", position=self.score_position + Vec(3, 1)))
        texts.append(
            Text(
                text=str(score),
                position=self.score_position + Vec(4, self._RIGHT_ELEMENTS_WIDTH - 1),
                alignment=Alignment.RIGHT,
            )
        )
        if self.total_num_games > 1:
            texts.append(Text(text="Rank", position=self.score_position + Vec(5, 1)))
            texts.append(
                Text(
                    text=f"{rank}/{self.total_num_games}",
                    position=self.score_position + Vec(6, self._RIGHT_ELEMENTS_WIDTH - 1),
                    alignment=Alignment.RIGHT,
                )
            )

    def _add_next_block_display(self, next_block: Block | None, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.next_block_position.y : self.next_block_position.y + self._NEXT_BLOCK_HEIGHT,
            self.next_block_position.x : self.next_block_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Next", position=self.next_block_position + Vec(1, 1)))

        if next_block is not None:
            next_block_ui_cells = next_block.actual_cells + ColorPalette.block_color_index_offset() - 1

            block_height, block_width = next_block_ui_cells.shape
            x_offset = ceil((self._RIGHT_ELEMENTS_WIDTH - 2 - block_width) / 2)
            y_offset = ceil((self._NEXT_BLOCK_HEIGHT - 3 - block_height) / 2)

            # handle powerup blocks (custom dynamic color, and with "?" text on top)
            powerup_positions = np.where(
                (next_block.actual_cells >= Board.MIN_POWERUP_CELL_VALUE)
                & (next_block.actual_cells <= Board.MAX_POWERUP_CELL_VALUE)
            )
            for y, x in zip(*powerup_positions, strict=True):
                next_block_ui_cells[y, x] = ColorPalette.DYNAMIC_POWERUP_INDEX
                texts.append(
                    Text(
                        text="?" * self.pixel_width,
                        position=self.next_block_position + Vec(2, 1) + Vec(y_offset, x_offset) + Vec(y, x),
                    )
                )

            # fmt: off
            np.copyto(
                ui_array[
                    self.next_block_position.y + 2 + y_offset : self.next_block_position.y + 2 + y_offset + block_height,  # noqa: E501
                    self.next_block_position.x + 1 + x_offset : self.next_block_position.x + 1 + x_offset + block_width,
                ],
                next_block_ui_cells,
                # using view() instead of astype() is an optimization that assumes that the int type of actual_cells is
                # 8 bit wide!
                where=next_block.actual_cells.view(bool),
            )
            # fmt: on

    def _add_level_display(self, level: int, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.level_position.y : self.level_position.y + self._LEVEL_HEIGHT,
            self.level_position.x : self.level_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Level", position=self.level_position + Vec(1, 1)))
        texts.append(
            Text(
                text=str(level),
                position=self.level_position + Vec(2, self._RIGHT_ELEMENTS_WIDTH - 1),
                alignment=Alignment.RIGHT,
            )
        )

    def _animation_overlays(self, animations: list[AnimationSpec]) -> list[Overlay]:
        overlay_animations: list[Overlay] = []
        for animation in animations:
            match animation:
                case TetrisAnimationSpec(
                    current_frame=current_frame,
                    total_frames=total_frames,
                    top_line_idx=top_line_idx,
                ):
                    overlay_animations.append(
                        Overlay(
                            position=self.board_position + Vec(top_line_idx, 0) + TetrisAnimationLeft.OFFSET,
                            frame=TetrisAnimationLeft.get_frame(current_frame, total_frames),
                        )
                    )
                    overlay_animations.append(
                        Overlay(
                            position=(
                                self.board_position + Vec(top_line_idx, self.board_width) + TetrisAnimationRight.OFFSET
                            ),
                            frame=TetrisAnimationRight.get_frame(current_frame, total_frames),
                        )
                    )
                case PowerupTriggeredAnimationSpec(
                    current_frame=current_frame, total_frames=total_frames, position=(y, x)
                ):
                    overlay_animations.append(
                        Overlay(
                            position=self.board_position + Vec(y, x) + PowerupTriggeredAnimation.OFFSET,
                            frame=PowerupTriggeredAnimation.get_frame(current_frame, total_frames),
                        )
                    )
                case BlooperAnimationSpec(current_frame=current_frame, total_frames=total_frames, seed=seed):
                    offset, frame = BlooperAnimation(
                        board_size=(self.board_height, self.board_width), seed=seed
                    ).get_offset_and_frame(current_frame=current_frame, total_frames=total_frames)

                    overlay_animations.append(Overlay(position=self.board_position + offset, frame=frame))
                case _:
                    msg = f"Unknown animation type: {type(animation)}"
                    raise ValueError(msg)

        return overlay_animations

    def _game_over_overlay(self, score: int, rank: int) -> Overlay:
        texts = [
            Text(
                text="GAME OVER",
                position=Vec(2, self._GAME_OVER_OVERLAY_WIDTH // 2),
                alignment=Alignment.CENTER,
            ),
            Text(
                text="Score:",
                position=Vec(4, 2),
                alignment=Alignment.LEFT,
            ),
            Text(
                text=str(score),
                position=Vec(4, self._GAME_OVER_OVERLAY_WIDTH - 2),
                alignment=Alignment.RIGHT,
            ),
        ]
        if self.total_num_games > 1:
            texts.append(
                Text(
                    text="Rank:",
                    position=Vec(5, 2),
                    alignment=Alignment.LEFT,
                )
            )
            texts.append(
                Text(
                    text=f"{rank}/{self.total_num_games}",
                    position=Vec(5, self._GAME_OVER_OVERLAY_WIDTH - 2),
                    alignment=Alignment.RIGHT,
                )
            )

        return Overlay(position=self.game_over_overlay_position, frame=self.game_over_overlay_frame, texts=texts)
