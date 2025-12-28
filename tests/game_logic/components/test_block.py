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


def test_unique_rotations() -> None:
    assert len(Block(BlockType.T).unique_rotations()) == 4  # noqa: PLR2004
    assert len(Block(BlockType.O).unique_rotations()) == 1
    assert len(Block(BlockType.I).unique_rotations()) == 2  # noqa: PLR2004
    assert len(Block(BlockType.L).unique_rotations()) == 4  # noqa: PLR2004
    assert len(Block(BlockType.S).unique_rotations()) == 2  # noqa: PLR2004
    assert len(Block(BlockType.J).unique_rotations()) == 4  # noqa: PLR2004
    assert len(Block(BlockType.Z).unique_rotations()) == 2  # noqa: PLR2004


def test_equals_in_shape() -> None:
    t1 = Block(BlockType.T)
    t2 = Block(BlockType.T)
    assert t1.equals_in_shape(t2)

    t2.rotate_right()
    assert t1.equals_in_shape(t2)

    o = Block(BlockType.O)
    assert not t1.equals_in_shape(o)

    s = Block(BlockType.S)
    s_rotated = Block(BlockType.S)
    s_rotated.rotate_right()
    assert s.equals_in_shape(s_rotated)

    z = Block(BlockType.Z)
    z_rotated = Block(BlockType.Z)
    z_rotated.rotate_right()
    assert z.equals_in_shape(z_rotated)

    assert not s.equals_in_shape(z)
