import pytest  # type: ignore
from game_logic.components import Block, BlockType, Board
from game_logic.components.exceptions import CannotDropBlock, CannotSpawnBlock, NoActiveBlock


def test_create_empty_board() -> None:
    board = Board.create_empty(10, 10)
    assert board.width == 10
    assert board.height == 10
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
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
        """
    )
    assert board.width == 10
    assert board.height == 12
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            ".....XXX..",
            "...XXX....",
            ".XXXXXXXXX",
            "XXXXXXXXX.",
            "XXXX...XXX",
        ]
    )


def test_board_from_string_representation_exceptions() -> None:
    with pytest.raises(ValueError):
        Board.from_string_representation(
            """
                ..........
                ...OOO....
                .XXXXXXXXX
                XXXXXXXXX.
            """
        )

    with pytest.raises(ValueError):
        Board.from_string_representation(
            """
                .............
                .......
                .XXXXXXXXX.
                XXXXXXXXX.
            """
        )


def test_spawn_block() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.I), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_spawn_block_at_position() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.S), position=(3, 2))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "..........",
            "..........",
            "...XX.....",
            "..XX......",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_spawn_block_partially_out_of_bounds_ok() -> None:
    board = Board.create_empty(10, 10)
    # the section of the I block that is out of bounds has no active cells
    board.spawn(Block(BlockType.I), position=(-2, 0))
    assert str(board) == "\n".join(
        [
            "XXXX......",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_spawn_block_partially_out_of_bounds_not_ok_but_can_be_nudged() -> None:
    board = Board.create_empty(10, 10)
    # the section of the I block that is out of bounds does have active cells, but it's close enough that it can be
    # nudged to be in bounds
    board.spawn(Block(BlockType.I), position=(0, -2))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_spawn_block_partially_out_of_bounds_not_ok() -> None:
    board = Board.create_empty(10, 10)
    with pytest.raises(CannotSpawnBlock):
        # the section of the I block that is out of bounds does have active cells (too far to be nudged)
        board.spawn(Block(BlockType.I), position=(0, -3))


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
        """
    )
    board.spawn(Block(BlockType.S), position=(0, 3))
    assert str(board) == "\n".join(
        [
            "..........",
            "....XX....",
            "...XXXXX..",
            "...XXX....",
            ".XXXXXXXXX",
            "XXXXXXXXX.",
            "XXXX...XXX",
        ]
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
        """
    )
    board.spawn(Block(BlockType.S), position=(0, 4))
    assert str(board) == "\n".join(
        [
            "..........",
            "....XX....",
            "...XXXXX..",
            "...XXX....",
            ".XXXXXXXXX",
            "XXXXXXXXX.",
            "XXXX...XXX",
        ]
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
        """
    )
    with pytest.raises(CannotSpawnBlock):
        board.spawn(Block(BlockType.S), position=(0, 5))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            ".....XXX..",
            "...XXX....",
            ".XXXXXXXXX",
            "XXXXXXXXX.",
            "XXXX...XXX",
        ]
    )


def test_spawn_block_top_middle_without_specified_position() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "..........",
            "..........",
        ]
    )

    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.I))
    assert str(board) == "\n".join(
        [
            "...XXXX...",
            "..........",
            "..........",
            "..........",
        ]
    )

    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.SQUARE))
    assert str(board) == "\n".join(
        [
            "....XX....",
            "....XX....",
            "..........",
            "..........",
        ]
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
        """
    )
    with pytest.raises(CannotSpawnBlock):
        board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "..........",
            "....XXXX..",
            "...XXX....",
            ".XXXXXXXXX",
            "XXXXXXXXX.",
            "XXXX...XXX",
        ]
    )


def test_drop_active_block() -> None:
    board = Board.create_empty(10, 10)
    board.spawn(Block(BlockType.I), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )
    board.drop_active_block()
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "..........",
            "XXXX......",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_drop_active_block_not_possible_bottom_of_board() -> None:
    board = Board.create_empty(3, 10)
    board.spawn(Block(BlockType.I), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
        ]
    )
    with pytest.raises(CannotDropBlock):
        board.drop_active_block()
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
        ]
    )


def test_drop_active_block_not_possible_hitting_active_cell_on_edge() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            .....X....
        """
    )
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            ".....X....",
        ]
    )
    with pytest.raises(CannotDropBlock):
        board.drop_active_block()
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            ".....X....",
        ]
    )


def test_drop_active_block_non_actual_bounding_box_overlapping_cell() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ..........
            ....X.....
            ....X.....
        """
    )
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "....X.....",
            "....X.....",
        ]
    )
    board.drop_active_block()
    assert str(board) == "\n".join(
        [
            "..........",
            "....XXX...",
            "....XX....",
            "....X.....",
        ]
    )


def test_drop_active_block_not_possible_hitting_active_cell_on_side() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ....X.....
            ....X.....
        """
    )
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            "....XX....",
            "....X.....",
        ]
    )
    with pytest.raises(CannotDropBlock):
        board.drop_active_block()
    assert str(board) == "\n".join(
        [
            "....XXX...",
            "....XX....",
            "....X.....",
        ]
    )


def test_rotate_block_left() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=(1, 1))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            ".XXX......",
            "..X.......",
        ]
    )
    board.try_rotate_active_block_left()
    assert str(board) == "\n".join(
        [
            "..........",
            "..X.......",
            "..XX......",
            "..X.......",
        ]
    )


def test_rotate_block_right() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=(1, 1))
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            ".XXX......",
            "..X.......",
        ]
    )
    board.try_rotate_active_block_right()
    assert str(board) == "\n".join(
        [
            "..........",
            "..X.......",
            ".XX.......",
            "..X.......",
        ]
    )


def test_rotate_block_beyond_top() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "..........",
            "..........",
        ]
    )
    board.try_rotate_active_block_right()
    assert str(board) == "\n".join(
        [
            "....XX....",
            ".....X....",
            "..........",
            "..........",
        ]
    )


def test_rotate_block_out_of_bounds_gets_nudged() -> None:
    board = Board.create_empty(4, 10)

    block = vertical_I_block()

    board.spawn(block, position=(0, -2))
    assert str(board) == "\n".join(
        [
            "X.........",
            "X.........",
            "X.........",
            "X.........",
        ]
    )
    board.try_rotate_active_block_right()
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "XXXX......",
            "..........",
        ]
    )


def vertical_I_block() -> Block:
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
        """
    )

    block = vertical_I_block()

    board.spawn(block, position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..X.......",
            "..XX......",
            "..XX......",
            "..XX......",
        ]
    )
    board.try_rotate_active_block_right()
    assert str(board) == "\n".join(
        [
            "..X.......",
            "..XX......",
            "..XX......",
            "..XX......",
        ]
    )


def test_rotate_block_into_other_blocks_gets_nudged() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ...X......
            ...X......
            ...X......
        """
    )

    block = Block(BlockType.I)
    block.rotate_left()

    board.spawn(block, position=(0, 2))
    assert str(board) == "\n".join(
        [
            "....X.....",
            "...XX.....",
            "...XX.....",
            "...XX.....",
        ]
    )
    board.try_rotate_active_block_right()
    assert str(board) == "\n".join(
        [
            "..........",
            "...X......",
            "...XXXXX..",
            "...X......",
        ]
    )


def test_move_right() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "..........",
            "..........",
        ]
    )

    board.try_move_active_block_right()

    assert str(board) == "\n".join(
        [
            ".....XXX..",
            "......X...",
            "..........",
            "..........",
        ]
    )


def test_move_left() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "..........",
            "..........",
        ]
    )

    board.try_move_active_block_left()

    assert str(board) == "\n".join(
        [
            "...XXX....",
            "....X.....",
            "..........",
            "..........",
        ]
    )


def test_move_not_possible_out_of_bounds() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "XXX.......",
            ".X........",
            "..........",
        ]
    )

    board.try_move_active_block_left()

    assert str(board) == "\n".join(
        [
            "..........",
            "XXX.......",
            ".X........",
            "..........",
        ]
    )


def test_move_not_possible_other_cells() -> None:
    board = Board.from_string_representation(
        """
            ...X......
            ...X......
            ...X......
            ...X......
        """
    )
    board.spawn(Block(BlockType.T), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "...X......",
            "XXXX......",
            ".X.X......",
            "...X......",
        ]
    )

    board.try_move_active_block_right()

    assert str(board) == "\n".join(
        [
            "...X......",
            "XXXX......",
            ".X.X......",
            "...X......",
        ]
    )


def test_merge_active_block_into_board() -> None:
    board = Board.create_empty(4, 10)
    board.spawn(Block(BlockType.T), position=(0, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "XXX.......",
            ".X........",
            "..........",
        ]
    )

    board.merge_active_block()
    assert_no_active_block_action_possible(board)

    board.spawn(Block(BlockType.T), position=(0, 3))
    assert str(board) == "\n".join(
        [
            "..........",
            "XXXXXX....",
            ".X..X.....",
            "..........",
        ]
    )


def assert_no_active_block_action_possible(board: Board) -> None:
    with pytest.raises(NoActiveBlock):
        board.try_move_active_block_left()

    with pytest.raises(NoActiveBlock):
        board.try_move_active_block_right()

    with pytest.raises(NoActiveBlock):
        board.try_rotate_active_block_left()

    with pytest.raises(NoActiveBlock):
        board.try_rotate_active_block_right()


def test_zero_lines_cleared() -> None:
    board = Board.create_empty(2, 10)
    board.spawn(Block(BlockType.I), position=(-1, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "XXXX......",
        ]
    )
    assert board.merge_active_block() == 0
    assert str(board) == "\n".join(
        [
            "..........",
            "XXXX......",
        ]
    )


def test_one_line_cleared() -> None:
    board = Board.from_string_representation(
        """
            ..........
            ....XXXXXX
        """
    )

    board.spawn(Block(BlockType.I), position=(-1, 0))
    assert str(board) == "\n".join(
        [
            "..........",
            "XXXXXXXXXX",
        ]
    )
    assert board.merge_active_block() == 1
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
        ]
    )


def test_four_lines_cleared() -> None:
    board = Board.from_string_representation(
        """
            .XXXXXXXXX
            .XXXXXXXXX
            .XXXXXXXXX
            .XXXXXXXXX
        """
    )

    board.spawn(vertical_I_block(), position=(0, -2))
    assert str(board) == "\n".join(
        [
            "XXXXXXXXXX",
            "XXXXXXXXXX",
            "XXXXXXXXXX",
            "XXXXXXXXXX",
        ]
    )
    assert board.merge_active_block() == 4
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "..........",
            "..........",
        ]
    )


def test_lines_above_clear_drop_down() -> None:
    board = Board.from_string_representation(
        """
            .........X
            ...XX....X
            .XXXXXXXXX
            .....XXXXX
        """
    )

    board.spawn(vertical_I_block(), position=(0, -2))
    assert str(board) == "\n".join(
        [
            "X........X",
            "X..XX....X",
            "XXXXXXXXXX",
            "X....XXXXX",
        ]
    )
    assert board.merge_active_block() == 1
    assert str(board) == "\n".join(
        [
            "..........",
            "X........X",
            "X..XX....X",
            "X....XXXXX",
        ]
    )


def test_lines_above_disconnected_line_clear_drop_down_correctly() -> None:
    board = Board.from_string_representation(
        """
            .........X
            .XXXXXXXXX
            .....XXXXX
            .XXXXXXXXX
        """
    )

    board.spawn(vertical_I_block(), position=(0, -2))
    assert str(board) == "\n".join(
        [
            "X........X",
            "XXXXXXXXXX",
            "X....XXXXX",
            "XXXXXXXXXX",
        ]
    )
    assert board.merge_active_block() == 2
    assert str(board) == "\n".join(
        [
            "..........",
            "..........",
            "X........X",
            "X....XXXXX",
        ]
    )
