import colorsys
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from math import ceil

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block
from tetris.game_logic.components.board import Board
from tetris.game_logic.interfaces.animations import AnimationSpec, TetrisAnimationSpec
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.ui.cli.animations import Overlay, TetrisAnimationLeft, TetrisAnimationRight
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


@dataclass(frozen=True)
class SingleGameUI:
    board_height: int
    board_width: int

    _rainbow_layer: NDArray[np.uint8] = field(init=False, repr=False, hash=False, compare=False)

    _RIGHT_GAP_WIDTH = 2
    _RIGHT_ELEMENTS_WIDTH = 6

    _LINES_HEIGHT = 3
    _CONTROLLER_SYMBOL_HEIGHT = 3
    _DISPLAY_GAP_HEIGHT = 1
    _SCORE_HEIGHT = 6
    _NEXT_BLOCK_HEIGHT = 5
    _LEVEL_HEIGHT = 4

    # What the UI looks like:

    #                      board_width       RIGHT_ELEMENTS_WIDTH
    #                  <------------------>    <---------->
    #
    #               ^  ____________________....____________  ^
    #  LINES_HEIGHT |  __Lines_________99__....____WASD____  | CONTROLLER_SYMBOL_HEIGHT
    #               v  ____________________....____________  v
    #                  ....................................    | DISPLAY_GAP_HEIGHT
    #               ^  ####################....____________  ^
    #               |  ####################....__Top_______  |
    #               |  ####################....__99999999__  | SCORE_HEIGHT
    #               |  ####################....__Score_____  |
    #               |  ####################...._____12345__  |
    #               |  ####################....____________  v
    #               |  ####################................    | DISPLAY_GAP_HEIGHT
    # board_height  |  ####################....____________  ^
    #               |  ####################....__Next______  |
    #               |  ####################....____##______  | NEXT_BLOCK_HEIGHT
    #               |  ####################....__######____  |
    #               |  ####################....____________  v
    #               |  ####################................    | DISPLAY_GAP_HEIGHT
    #               |  ####################....____________  ^
    #               |  ####################....__Level_____  | LEVEL_HEIGHT
    #               |  ####################...._________9__  |
    #               v  ####################....____________  v
    #
    #                                      <-->
    #                                 RIGHT_GAP_WIDTH

    # legend:
    # # board/block
    # _ display background (black)
    # . gap (mask=False, filled by outer background)
    # note: 2 characters constitute one pixel

    def __post_init__(self) -> None:
        height, width = self.total_size
        # an array where values increase diagonally to be used for rainbow coloring
        object.__setattr__(
            self, "_rainbow_layer", np.add.outer(np.arange(height, dtype=np.uint8), np.arange(width, dtype=np.uint8))
        )

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
                self._LINES_HEIGHT + self._DISPLAY_GAP_HEIGHT + self.board_height,
                (
                    self._CONTROLLER_SYMBOL_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._SCORE_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._NEXT_BLOCK_HEIGHT
                    + self._DISPLAY_GAP_HEIGHT
                    + self._LEVEL_HEIGHT
                ),
            ),
            self.board_width + self._RIGHT_GAP_WIDTH + self._RIGHT_ELEMENTS_WIDTH,
        )

    @cached_property
    def lines_position(self) -> Vec:
        return Vec(0, 0)

    @cached_property
    def board_position(self) -> Vec:
        return Vec(self._LINES_HEIGHT + self._DISPLAY_GAP_HEIGHT, 0)

    @cached_property
    def controller_symbol_position(self) -> Vec:
        return Vec(
            0,
            self.board_width + self._RIGHT_GAP_WIDTH,
        )

    @cached_property
    def score_position(self) -> Vec:
        return Vec(
            self._CONTROLLER_SYMBOL_HEIGHT + self._DISPLAY_GAP_HEIGHT,
            self.board_width + self._RIGHT_GAP_WIDTH,
        )

    @cached_property
    def next_block_position(self) -> Vec:
        return Vec(
            self._CONTROLLER_SYMBOL_HEIGHT + self._DISPLAY_GAP_HEIGHT + self._SCORE_HEIGHT + self._DISPLAY_GAP_HEIGHT,
            self.board_width + self._RIGHT_GAP_WIDTH,
        )

    @cached_property
    def level_position(self) -> Vec:
        return Vec(
            (
                self._CONTROLLER_SYMBOL_HEIGHT
                + self._DISPLAY_GAP_HEIGHT
                + self._SCORE_HEIGHT
                + self._DISPLAY_GAP_HEIGHT
                + self._NEXT_BLOCK_HEIGHT
                + self._DISPLAY_GAP_HEIGHT
            ),
            self.board_width + self._RIGHT_GAP_WIDTH,
        )

    @cached_property
    def mask(self) -> NDArray[np.bool]:
        mask = np.zeros(self.total_size, dtype=np.bool)

        mask[
            self.lines_position.y : self.lines_position.y + self._LINES_HEIGHT,
            self.lines_position.x : self.lines_position.x + self.board_width,
        ] = True
        mask[
            self.board_position.y : self.board_position.y + self.board_height,
            self.board_position.x : self.board_position.x + self.board_width,
        ] = True
        mask[
            self.controller_symbol_position.y : self.controller_symbol_position.y + self._CONTROLLER_SYMBOL_HEIGHT,
            self.controller_symbol_position.x : self.controller_symbol_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.score_position.y : self.score_position.y + self._SCORE_HEIGHT,
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
        self._add_board(board=elements.board, ui_array=ui_array)
        self._add_controller_symbol_display(
            controller_symbol=elements.controller_symbol, ui_array=ui_array, texts=texts
        )
        self._add_score_display(
            score=elements.score, session_high_score=elements.session_high_score, ui_array=ui_array, texts=texts
        )
        self._add_next_block_display(next_block=elements.next_block, ui_array=ui_array, texts=texts)
        self._add_level_display(level=elements.level, ui_array=ui_array, texts=texts)
        overlay_animations = self._add_animations(elements.animations)

        return ui_array, texts, overlay_animations

    def _add_lines_display(self, num_cleared_lines: int, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.lines_position.y : self.lines_position.y + self._LINES_HEIGHT,
            self.lines_position.x : self.lines_position.x + self.board_width,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Lines", position=self.lines_position + Vec(1, 1)))
        texts.append(
            Text(
                text=str(num_cleared_lines),
                position=self.lines_position + Vec(1, self.board_width - 1),
                alignment=Alignment.RIGHT,
            )
        )

    def _add_board(self, board: NDArray[np.uint8], ui_array: NDArray[np.uint8]) -> None:
        board_in_ui_array = np.where(board, board + ColorPalette.block_color_index_offset() - 1, self.board_background)

        if Board.GHOST_BLOCK_CELL_VALUE in board:
            # use ghost background color for ghost cells
            board_in_ui_array = np.where(
                board == Board.GHOST_BLOCK_CELL_VALUE,
                self.board_background
                + (ColorPalette.board_bg_ghost_index_offset() - ColorPalette.board_bg_index_offset()),
                board_in_ui_array,
            )

        ui_array[
            self.board_position.y : self.board_position.y + self.board_height,
            self.board_position.x : self.board_position.x + self.board_width,
        ] = board_in_ui_array

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
        self, score: int, session_high_score: int, ui_array: NDArray[np.uint8], texts: list[Text]
    ) -> None:
        # ui_array[
        #     self.score_position.y : self.score_position.y + self._SCORE_HEIGHT,
        #     self.score_position.x : self.score_position.x + self._RIGHT_ELEMENTS_WIDTH,
        # ] = ColorPalette.RAINBOW_INDEX
        ui_array[
            self.score_position.y : self.score_position.y + self._SCORE_HEIGHT,
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

    def _add_next_block_display(self, next_block: Block | None, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.next_block_position.y : self.next_block_position.y + self._NEXT_BLOCK_HEIGHT,
            self.next_block_position.x : self.next_block_position.x + self._RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Next", position=self.next_block_position + Vec(1, 1)))

        if next_block is not None:
            block_height, block_width = next_block.actual_cells.shape
            x_offset = ceil((self._RIGHT_ELEMENTS_WIDTH - 2 - block_width) / 2)
            y_offset = ceil((self._NEXT_BLOCK_HEIGHT - 3 - block_height) / 2)
            # fmt: off
            np.copyto(
                ui_array[
                    self.next_block_position.y + 2 + y_offset : self.next_block_position.y + 2 + y_offset + block_height,  # noqa: E501
                    self.next_block_position.x + 1 + x_offset : self.next_block_position.x + 1 + x_offset + block_width,
                ],
                next_block.actual_cells + ColorPalette.block_color_index_offset() - 1,
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

    def _add_animations(self, animations: list[AnimationSpec]) -> list[Overlay]:
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
                case _:
                    msg = f"Unknown animation type: {type(animation)}"
                    raise ValueError(msg)

        return overlay_animations

    @staticmethod
    def _get_rainbow_color(draw_counter: int, position: Vec) -> str:
        hue = (draw_counter * 5 + position.y * 10 + position.x * 10) % 360 / 360.0
        saturation = 1.0
        value = 1.0

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return ColorPalette.color_fn(*(round(c * 255) for c in rgb))
        # Create a rainbow effect that changes over time and position
        base_hue = (draw_counter * 5 + position.y * 10 + position.x * 10) % 360
        return ColorPalette.index_of_color(
            f"rainbow_{base_hue // 15 * 15}"
        )  # assuming colors are defined every 15 degrees
