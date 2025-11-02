import re

import numpy as np
import pytest

from tetris.game_logic.components import Block, BlockType, Board
from tetris.game_logic.components.block import Vec
from tetris.game_logic.components.exceptions import (
    ActiveBlockOverlapError,
    CannotDropBlockError,
    CannotSpawnBlockError,
    NoActiveBlockError,
)


def test_create_empty_board() -> None:
    h, w = 10, 10
    board = Board.create_empty(h, w)
    assert board.height == h
    assert board.width == w
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_board_from_string_representation() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """,
    )
    assert board.height == 12  # noqa: PLR2004
    assert board.width == 10  # noqa: PLR2004
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """.replace(" ", "").strip()
    )


def test_board_from_string_representation_exceptions() -> None:
    with pytest.raises(ValueError, match=re.escape("Must consist of only 'X', '.', spaces, and newlines")):
        Board.from_string_representation(
            """
                ..........
                ...OOO....
                .XXXXXXXXX
                XXXXXXXXX.
            """,
        )

    with pytest.raises(ValueError, match="all lines must have the same width"):
        Board.from_string_representation(
            """
                .............
                .......
                .XXXXXXXXX.
                XXXXXXXXX.
            """,
        )


def test_spawn_block() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.I), position=Vec(0, 0))
    assert (
        str(board)
        == """
        ..........
        ..........
        XXXX......
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        ..........
        """.replace(" ", "").strip()
    )


def test_spawn_block_at_position() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.S), position=Vec(3, 2))
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            ..........
            ...XX.....
            ..XX......
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_spawn_block_partially_out_of_bounds_ok() -> None:
    board = Board.create_empty(10, 10)
    # the section of the I block that is out of bounds has no active cells
    board.spawn(Block(BlockType.I), position=Vec(-2, 0))
    assert (
        str(board)
        == """
            XXXX......
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_spawn_block_partially_out_of_bounds_not_ok_but_can_be_nudged() -> None:
    board = Board.create_empty(10, 10)
    # the section of the I block that is out of bounds does have active cells, but it's close enough that it can be
    # nudged to be in bounds
    board.spawn(Block(BlockType.I), position=Vec(0, -2))
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXX......
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_spawn_block_partially_out_of_bounds_not_ok() -> None:
    board = Board.create_empty(10, 10)
    with pytest.raises(CannotSpawnBlockError):
        # the section of the I block that is out of bounds does have active cells (too far to be nudged)
        board.spawn(Block(BlockType.I), position=Vec(0, -3))


def test_spawn_block_not_overlapping_existing_cells() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """,
    )
    board.spawn(Block(BlockType.S), position=Vec(0, 3))
    assert (
        str(board)
        == """
            ..........
            ....XX....
            ...XXXXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """.replace(" ", "").strip()
    )


def test_spawn_block_overlapping_existing_cells_but_can_be_nudged() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """,
    )
    board.spawn(Block(BlockType.S), position=Vec(0, 4))
    assert (
        str(board)
        == """
            ..........
            ....XX....
            ...XXXXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """.replace(" ", "").strip()
    )


def test_spawn_block_overlapping_existing_cells() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """,
    )
    with pytest.raises(CannotSpawnBlockError):
        board.spawn(Block(BlockType.S), position=Vec(0, 5))
    assert (
        str(board)
        == """
            ..........
            ..........
            .....XXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """.replace(" ", "").strip()
    )


def test_spawn_block_top_middle_without_specified_position() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            ..........
            ..........
        """.replace(" ", "").strip()
    )

    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.I))
    assert (
        str(board)
        == """
            ...XXXX...
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )

    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.O))
    assert (
        str(board)
        == """
            ....XX....
            ....XX....
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_spawn_block_top_middle_without_specified_position_overlapping_existing_blocks() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ....XXXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """,
    )
    with pytest.raises(CannotSpawnBlockError):
        board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ..........
            ....XXXX..
            ...XXX....
            .XXXXXXXXX
            XXXXXXXXX.
            XXXX...XXX
        """.replace(" ", "").strip()
    )


def test_drop_active_block() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.I), position=Vec(0, 0))
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXX......
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )
    board.drop_active_block()
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            XXXX......
            ..........
            ..........
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_drop_active_block_not_possible_bottom_of_board() -> None:
    board = Board.create_empty(3, 10)
    board.spawn(Block(BlockType.I), position=Vec(0, 0))
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXX......
        """.replace(" ", "").strip()
    )
    with pytest.raises(CannotDropBlockError):
        board.drop_active_block()
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXX......
        """.replace(" ", "").strip()
    )


def test_drop_active_block_not_possible_hitting_active_cell_on_edge() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            .....X....
        """,
    )
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            .....X....
        """.replace(" ", "").strip()
    )
    with pytest.raises(CannotDropBlockError):
        board.drop_active_block()
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            .....X....
        """.replace(" ", "").strip()
    )


def test_drop_active_block_non_actual_bounding_box_overlapping_cell() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            ....X.....
            ....X.....
        """,
    )
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            ....X.....
            ....X.....
        """.replace(" ", "").strip()
    )
    board.drop_active_block()
    assert (
        str(board)
        == """
            ..........
            ....XXX...
            ....XX....
            ....X.....
        """.replace(" ", "").strip()
    )


def test_drop_active_block_not_possible_hitting_active_cell_on_side() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ....X.....
            ....X.....
        """,
    )
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            ....XX....
            ....X.....
        """.replace(" ", "").strip()
    )
    with pytest.raises(CannotDropBlockError):
        board.drop_active_block()
    assert (
        str(board)
        == """
            ....XXX...
            ....XX....
            ....X.....
        """.replace(" ", "").strip()
    )


def test_rotate_block_left() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=Vec(1, 1))
    assert (
        str(board)
        == """
            ..........
            ..........
            .XXX......
            ..X.......
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_left()
    assert (
        str(board)
        == """
            ..........
            ..X.......
            ..XX......
            ..X.......
        """.replace(" ", "").strip()
    )


def test_rotate_block_right() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=Vec(1, 1))
    assert (
        str(board)
        == """
            ..........
            ..........
            .XXX......
            ..X.......
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_right()
    assert (
        str(board)
        == """
            ..........
            ..X.......
            .XX.......
            ..X.......
        """.replace(" ", "").strip()
    )


def test_rotate_block_beyond_top() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            ..........
            ..........
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_right()
    assert (
        str(board)
        == """
            ....XX....
            .....X....
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_rotate_block_out_of_bounds_gets_nudged() -> None:
    board = Board.create_empty(4, 10)

    block = vertical_I_block()

    board.spawn(block, position=Vec(0, -2))
    assert (
        str(board)
        == """
            X.........
            X.........
            X.........
            X.........
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_right()
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXX......
            ..........
        """.replace(" ", "").strip()
    )


def vertical_I_block() -> Block:  # noqa: N802
    block = Block(BlockType.I)
    block.rotate_left()
    return block


def test_rotate_block_into_other_blocks_nudging_fails() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ...X......
            ...X......
            ...X......
        """,
    )

    block = vertical_I_block()

    board.spawn(block, position=Vec(0, 0))
    assert (
        str(board)
        == """
            ..X.......
            ..XX......
            ..XX......
            ..XX......
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_right()
    assert (
        str(board)
        == """
            ..X.......
            ..XX......
            ..XX......
            ..XX......
        """.replace(" ", "").strip()
    )


def test_rotate_block_into_other_blocks_gets_nudged() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ...X......
            ...X......
            ...X......
        """,
    )

    block = Block(BlockType.I)
    block.rotate_left()

    board.spawn(block, position=Vec(0, 2))
    assert (
        str(board)
        == """
            ....X.....
            ...XX.....
            ...XX.....
            ...XX.....
        """.replace(" ", "").strip()
    )
    board.try_rotate_active_block_right()
    assert (
        str(board)
        == """
            ..........
            ...X......
            ...XXXXX..
            ...X......
        """.replace(" ", "").strip()
    )


def test_move_right() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            ..........
            ..........
        """.replace(" ", "").strip()
    )

    board.try_move_active_block_right()

    assert (
        str(board)
        == """
            .....XXX..
            ......X...
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_move_left() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert (
        str(board)
        == """
            ....XXX...
            .....X....
            ..........
            ..........
        """.replace(" ", "").strip()
    )

    board.try_move_active_block_left()

    assert (
        str(board)
        == """
            ...XXX....
            ....X.....
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_move_not_possible_out_of_bounds() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=Vec(0, 0))
    assert (
        str(board)
        == """
            ..........
            XXX.......
            .X........
            ..........
        """.replace(" ", "").strip()
    )

    board.try_move_active_block_left()

    assert (
        str(board)
        == """
            ..........
            XXX.......
            .X........
            ..........
        """.replace(" ", "").strip()
    )


def test_move_not_possible_other_cells() -> None:
    board = Board.from_string_representation(
        """
            ...X......
            ...X......
            ...X......
            ...X......
        """,
    )
    board.spawn(Block(BlockType.T), position=Vec(0, 0))
    assert (
        str(board)
        == """
            ...X......
            XXXX......
            .X.X......
            ...X......
        """.replace(" ", "").strip()
    )

    board.try_move_active_block_right()

    assert (
        str(board)
        == """
            ...X......
            XXXX......
            .X.X......
            ...X......
        """.replace(" ", "").strip()
    )


def test_merge_active_block_into_board() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=Vec(0, 0))
    assert (
        str(board)
        == """
            ..........
            XXX.......
            .X........
            ..........
        """.replace(" ", "").strip()
    )

    board.merge_active_block()
    assert_no_active_block_action_possible(board)

    board.spawn(Block(BlockType.T), position=Vec(0, 3))
    assert (
        str(board)
        == """
            ..........
            XXXXXX....
            .X..X.....
            ..........
        """.replace(" ", "").strip()
    )


def assert_no_active_block_action_possible(board: Board) -> None:
    with pytest.raises(NoActiveBlockError):
        board.try_move_active_block_left()

    with pytest.raises(NoActiveBlockError):
        board.try_move_active_block_right()

    with pytest.raises(NoActiveBlockError):
        board.try_rotate_active_block_left()

    with pytest.raises(NoActiveBlockError):
        board.try_rotate_active_block_right()


def test_zero_lines_cleared() -> None:
    board = Board.create_empty(2, 10)
    board.spawn(Block(BlockType.I), position=Vec(-1, 0))
    assert (
        str(board)
        == """
            ..........
            XXXX......
        """.replace(" ", "").strip()
    )
    board.merge_active_block()
    assert (
        str(board)
        == """
            ..........
            XXXX......
        """.replace(" ", "").strip()
    )


def test_get_full_line_idxs_one_line() -> None:
    board = Board.from_string_representation(
        """
            ..........
            XXXXXXXXXX
        """,
    )

    assert board.get_full_line_idxs() == [1]


def test_get_full_line_idxs_four_lines() -> None:
    board = Board.from_string_representation(
        """
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
        """,
    )

    assert board.get_full_line_idxs() == [0, 1, 2, 3]


def test_get_full_line_idxs_partial_lines() -> None:
    board = Board.from_string_representation(
        """
            X........X
            X..XX....X
            XXXXXXXXXX
            X....XXXXX
        """,
    )

    assert board.get_full_line_idxs() == [2]


def test_get_full_line_idxs_disconnected_lines() -> None:
    board = Board.from_string_representation(
        """
            X........X
            XXXXXXXXXX
            X....XXXXX
            XXXXXXXXXX
        """,
    )

    assert board.get_full_line_idxs() == [1, 3]


def test_clear_line() -> None:
    board = Board.from_string_representation(
        """
            ..........
            XXXXXXXXXX
        """,
    )

    board.clear_line(1)
    assert (
        str(board)
        == """
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_clear_lines_four_lines() -> None:
    board = Board.from_string_representation(
        """
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
        """,
    )

    board.clear_lines([0, 1, 2, 3])
    assert (
        str(board)
        == """
            ..........
            ..........
            ..........
            ..........
        """.replace(" ", "").strip()
    )


def test_clear_lines_partial_lines() -> None:
    board = Board.from_string_representation(
        """
            X........X
            X..XX....X
            XXXXXXXXXX
            X....XXXXX
        """,
    )

    board.clear_lines([2, 3])
    assert (
        str(board)
        == """
            ..........
            ..........
            X........X
            X..XX....X
        """.replace(" ", "").strip()
    )


def test_clear_lines_disconnected_lines() -> None:
    board = Board.from_string_representation(
        """
            X........X
            XXXXXXXXXX
            X....XXXXX
            XXXXXXXXXX
        """,
    )

    board.clear_lines([0, 2])
    assert (
        str(board)
        == """
            ..........
            ..........
            XXXXXXXXXX
            XXXXXXXXXX
        """.replace(" ", "").strip()
    )


def test_clear_line_invalid_idx() -> None:
    board = Board.from_string_representation(
        """
            XXXXXXXXXX
            XXXXXXXXXX
            XXXXXXXXXX
        """,
    )

    with pytest.raises(IndexError):
        board.clear_line(3)

    with pytest.raises(IndexError):
        board.clear_line(-1)


def test_set_from_array() -> None:
    board = Board.create_empty(5, 5)
    board.set_from_array(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.bool,
        ),
    )
    assert (
        str(board)
        == """
            .....
            .....
            ..X..
            .X.X.
            XXXXX
        """.replace(" ", "").strip()
    )


def test_set_from_array_overlapping_active_block() -> None:
    board = Board.create_empty(5, 5)
    board.spawn(Block(BlockType.T), position=Vec(0, 0))
    assert (
        str(board)
        == """
            .....
            XXX..
            .X...
            .....
            .....
        """.replace(" ", "").strip()
    )

    with pytest.raises(ActiveBlockOverlapError):
        board.set_from_array(
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
                dtype=np.bool,
            ),
        )
