from tetris.exceptions import BaseTetrisError


class NoActiveBlockError(BaseTetrisError):
    pass


class CannotSpawnBlockError(BaseTetrisError):
    pass


class CannotDropBlockError(BaseTetrisError):
    pass


class CannotNudgeError(BaseTetrisError):
    pass


class ActiveBlockOverlapError(BaseTetrisError):
    pass
