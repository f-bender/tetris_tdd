from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from math import ceil

import numpy as np
from numpy.typing import NDArray

from tetris.game_logic.components.block import Block
from tetris.game_logic.interfaces.animations import AnimationSpec, TetrisAnimationSpec
from tetris.game_logic.interfaces.ui import SingleUiElements
from tetris.ui.cli.animations import Overlay, TetrisAnimationLeft, TetrisAnimationRight
from tetris.ui.cli.color_palette import ColorPalette
from tetris.ui.cli.vec import Vec


class Alignment(Enum):
    LEFT = auto()
    RIGHT = auto()
    CENTER = auto()


_DISPLAY_BG_COLOR_INDEX = ColorPalette.index_of_color("display_bg")


@dataclass(unsafe_hash=True, slots=True)
class Text:
    text: str
    position: Vec
    alignment: Alignment = Alignment.LEFT
    bg_color_index: int = _DISPLAY_BG_COLOR_INDEX


@dataclass(frozen=True)
class SingleGameUI:
    board_background: NDArray[np.uint8]

    RIGHT_GAP_WIDTH = 4
    RIGHT_ELEMENTS_WIDTH = 6

    DISPLAY_GAP_HEIGHT = 1
    SCORE_HEIGHT = 4
    NEXT_BLOCK_HEIGHT = 5
    CONTROLLER_SYMBOL_HEIGHT = 3

    # What the UI looks like:

    #                 board_size[1] (width)      RIGHT_ELEMENTS_WIDTH
    #                  <------------------>        <---------->
    #
    #               ^  ####################........____________  ^
    #               |  ####################........____WASD____  | CONTROLLER_SYMBOL_HEIGHT
    #               |  ####################........____________  v
    #               |  ####################....................    | DISPLAY_GAP_HEIGHT
    #               |  ####################........____________  ^
    #               |  ####################........__Score_____  | SCORE_HEIGHT
    #               |  ####################........__99999999__  |
    #               |  ####################........____________  v
    # board_size[0] |  ####################....................    | DISPLAY_GAP_HEIGHT
    # (height)      |  ####################........____________  ^
    #               |  ####################........__Next______  |
    #               |  ####################........____##______  | NEXT_BLOCK_HEIGHT
    #               |  ####################........__######____  |
    #               v  ####################........____________  v
    #
    #                                      <------>
    #                                   RIGHT_GAP_WIDTH

    # legend:
    # # board/block
    # _ display background (black)
    # . gap (mask=False, filled by outer background)
    # note: 2 characters constitute one pixel

    @cached_property
    def board_size(self) -> tuple[int, int]:
        return tuple(self.board_background.shape)

    @cached_property
    def total_size(self) -> tuple[int, int]:
        return (
            max(self.board_size[0], self.SCORE_HEIGHT + self.DISPLAY_GAP_HEIGHT + self.NEXT_BLOCK_HEIGHT),
            self.board_size[1] + self.RIGHT_GAP_WIDTH + self.RIGHT_ELEMENTS_WIDTH,
        )

    @cached_property
    def controller_symbol_display_position(self) -> Vec:
        return Vec(
            0,
            self.board_size[1] + self.RIGHT_GAP_WIDTH,
        )

    @cached_property
    def score_display_position(self) -> Vec:
        return Vec(
            self.CONTROLLER_SYMBOL_HEIGHT + self.DISPLAY_GAP_HEIGHT,
            self.board_size[1] + self.RIGHT_GAP_WIDTH,
        )

    @cached_property
    def next_block_position(self) -> Vec:
        return Vec(
            self.CONTROLLER_SYMBOL_HEIGHT + self.DISPLAY_GAP_HEIGHT + self.SCORE_HEIGHT + self.DISPLAY_GAP_HEIGHT,
            self.board_size[1] + self.RIGHT_GAP_WIDTH,
        )

    @property
    def mask(self) -> NDArray[np.bool]:
        mask = np.zeros(self.total_size, dtype=np.bool)

        mask[: self.board_size[0], : self.board_size[1]] = True
        mask[
            self.controller_symbol_display_position.y : self.controller_symbol_display_position.y
            + self.CONTROLLER_SYMBOL_HEIGHT,
            self.controller_symbol_display_position.x : self.controller_symbol_display_position.x
            + self.RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.score_display_position.y : self.score_display_position.y + self.SCORE_HEIGHT,
            self.score_display_position.x : self.score_display_position.x + self.RIGHT_ELEMENTS_WIDTH,
        ] = True
        mask[
            self.next_block_position.y : self.next_block_position.y + self.NEXT_BLOCK_HEIGHT,
            self.next_block_position.x : self.next_block_position.x + self.RIGHT_ELEMENTS_WIDTH,
        ] = True

        return mask

    def create_array_texts_animations(
        self, elements: SingleUiElements
    ) -> tuple[NDArray[np.uint8], list[Text], list[Overlay]]:
        ui_array = np.empty(self.total_size, dtype=np.uint8)
        texts: list[Text] = []

        self._add_board(board=elements.board, ui_array=ui_array)
        self._add_controller_symbol_display(
            controller_symbol=elements.controller_symbol, ui_array=ui_array, texts=texts
        )
        self._add_score_display(score=elements.score, ui_array=ui_array, texts=texts)
        self._add_next_block_display(next_block=elements.next_block, ui_array=ui_array, texts=texts)
        overlay_animations = self._add_animations(elements.animations)

        return ui_array, texts, overlay_animations

    def _add_board(self, board: NDArray[np.uint8], ui_array: NDArray[np.uint8]) -> None:
        ui_array[: self.board_size[0], : self.board_size[1]] = np.where(
            board, board + ColorPalette.block_color_index_offset() - 1, self.board_background
        )

    def _add_controller_symbol_display(
        self, controller_symbol: str, ui_array: NDArray[np.uint8], texts: list[Text]
    ) -> None:
        ui_array[
            self.controller_symbol_display_position.y : self.controller_symbol_display_position.y
            + self.CONTROLLER_SYMBOL_HEIGHT,
            self.controller_symbol_display_position.x : self.controller_symbol_display_position.x
            + self.RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(
            Text(
                text=controller_symbol,
                position=self.controller_symbol_display_position + Vec(1, self.RIGHT_ELEMENTS_WIDTH // 2),
                alignment=Alignment.CENTER,
            )
        )

    def _add_score_display(self, score: int, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.score_display_position.y : self.score_display_position.y + self.SCORE_HEIGHT,
            self.score_display_position.x : self.score_display_position.x + self.RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Score", position=self.score_display_position + Vec(1, 1)))
        texts.append(
            Text(
                text=str(score),
                position=self.score_display_position + Vec(2, self.RIGHT_ELEMENTS_WIDTH - 1),
                alignment=Alignment.RIGHT,
            )
        )

    def _add_next_block_display(self, next_block: Block | None, ui_array: NDArray[np.uint8], texts: list[Text]) -> None:
        ui_array[
            self.next_block_position.y : self.next_block_position.y + self.NEXT_BLOCK_HEIGHT,
            self.next_block_position.x : self.next_block_position.x + self.RIGHT_ELEMENTS_WIDTH,
        ] = ColorPalette.index_of_color("display_bg")
        texts.append(Text(text="Next", position=self.next_block_position + Vec(1, 1)))

        if next_block is not None:
            block_height, block_width = next_block.actual_cells.shape
            x_offset = ceil((self.RIGHT_ELEMENTS_WIDTH - 2 - block_width) / 2)
            y_offset = ceil((self.NEXT_BLOCK_HEIGHT - 3 - block_height) / 2)
            # fmt: off
            np.copyto(
                ui_array[
                    self.next_block_position.y + 2 + y_offset : self.next_block_position.y + 2 + y_offset + block_height,  # noqa: E501
                    self.next_block_position.x + 1 + x_offset : self.next_block_position.x + 1 + x_offset + block_width,
                ],
                next_block.actual_cells + ColorPalette.block_color_index_offset() - 1,
                where=next_block.actual_cells.astype(bool),
            )
            # fmt: on

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
                            position=Vec(top_line_idx, 0) + TetrisAnimationLeft.OFFSET,
                            frame=TetrisAnimationLeft.get_frame(current_frame, total_frames),
                        )
                    )
                    overlay_animations.append(
                        Overlay(
                            position=Vec(top_line_idx, self.board_size[1]) + TetrisAnimationRight.OFFSET,
                            frame=TetrisAnimationRight.get_frame(current_frame, total_frames),
                        )
                    )
                case _:
                    msg = f"Unknown animation type: {type(animation)}"
                    raise ValueError(msg)

        return overlay_animations
