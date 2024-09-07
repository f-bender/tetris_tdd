from tetris.game_logic.components import Block, BlockType


def test_block_creation() -> None:
    Block(BlockType.O)


def test_block_representation() -> None:
    assert (
        str(Block(BlockType.O))
        == """
            XX
            XX
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.L))
        == """
            ...
            XXX
            X..
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.J))
        == """
            ...
            XXX
            ..X
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.Z))
        == """
            ...
            XX.
            .XX
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.S))
        == """
            ...
            .XX
            XX.
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.T))
        == """
            ...
            XXX
            .X.
        """.replace(" ", "").strip()
    )
    assert (
        str(Block(BlockType.I))
        == """
            ....
            ....
            XXXX
            ....
        """.replace(" ", "").strip()
    )


def test_block_rotation() -> None:
    s = Block(BlockType.S)
    s.rotate_left()
    assert (
        str(s)
        == """
            .X.
            .XX
            ..X
        """.replace(" ", "").strip()
    )
    s.rotate_left()
    assert (
        str(s)
        == """
            .XX
            XX.
            ...
        """.replace(" ", "").strip()
    )
    s.rotate_left()
    assert (
        str(s)
        == """
            X..
            XX.
            .X.
        """.replace(" ", "").strip()
    )

    i = Block(BlockType.I)
    i.rotate_right()
    assert (
        str(i)
        == """
            .X..
            .X..
            .X..
            .X..
        """.replace(" ", "").strip()
    )
    i.rotate_right()
    assert (
        str(i)
        == """
            ....
            XXXX
            ....
            ....
        """.replace(" ", "").strip()
    )
    i.rotate_right()
    assert (
        str(i)
        == """
            ..X.
            ..X.
            ..X.
            ..X.
        """.replace(" ", "").strip()
    )
