import pytest  # type: ignore

from game_logic.block import Block, BlockType
from game_logic.board import Board


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
    board = Board.empty(10, 10)
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
    board = Board.empty(10, 10)
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
    board = Board.empty(10, 10)
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


def test_spawn_block_partially_out_of_bounds_not_ok() -> None:
    board = Board.empty(10, 10)
    with pytest.raises(ValueError):
        # the section of the I block that is out of bounds does have active cells
        board.spawn(Block(BlockType.I), position=(-3, 0))


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
    with pytest.raises(ValueError):
        board.spawn(Block(BlockType.S), position=(0, 4))
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
    board = Board.empty(4, 10)
    board.spawn(Block(BlockType.T))
    assert str(board) == "\n".join(
        [
            "....XXX...",
            ".....X....",
            "..........",
            "..........",
        ]
    )

    board = Board.empty(4, 10)
    board.spawn(Block(BlockType.I))
    assert str(board) == "\n".join(
        [
            "...XXXX...",
            "..........",
            "..........",
            "..........",
        ]
    )

    board = Board.empty(4, 10)
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
    with pytest.raises(ValueError):
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
