from tetris.game_logic.components import Block, BlockType


def test_block_creation() -> None:
    Block(BlockType.O)


def test_block_representation() -> None:
    assert str(Block(BlockType.O)) == "\n".join(
        [
            "XX",
            "XX",
        ],
    )
    assert str(Block(BlockType.L)) == "\n".join(
        [
            "...",
            "XXX",
            "X..",
        ],
    )
    assert str(Block(BlockType.J)) == "\n".join(
        [
            "...",
            "XXX",
            "..X",
        ],
    )
    assert str(Block(BlockType.Z)) == "\n".join(
        [
            "...",
            "XX.",
            ".XX",
        ],
    )
    assert str(Block(BlockType.S)) == "\n".join(
        [
            "...",
            ".XX",
            "XX.",
        ],
    )
    assert str(Block(BlockType.T)) == "\n".join(
        [
            "...",
            "XXX",
            ".X.",
        ],
    )
    assert str(Block(BlockType.I)) == "\n".join(
        [
            "....",
            "....",
            "XXXX",
            "....",
        ],
    )


def test_block_rotation() -> None:
    s = Block(BlockType.S)
    s.rotate_left()
    assert str(s) == "\n".join(
        [
            ".X.",
            ".XX",
            "..X",
        ],
    )
    s.rotate_left()
    assert str(s) == "\n".join(
        [
            ".XX",
            "XX.",
            "...",
        ],
    )
    s.rotate_left()
    assert str(s) == "\n".join(
        [
            "X..",
            "XX.",
            ".X.",
        ],
    )

    i = Block(BlockType.I)
    i.rotate_right()
    assert str(i) == "\n".join(
        [
            ".X..",
            ".X..",
            ".X..",
            ".X..",
        ],
    )
    i.rotate_right()
    assert str(i) == "\n".join(
        [
            "....",
            "XXXX",
            "....",
            "....",
        ],
    )
    i.rotate_right()
    assert str(i) == "\n".join(
        [
            "..X.",
            "..X.",
            "..X.",
            "..X.",
        ],
    )
