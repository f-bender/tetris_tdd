class BaseTetrisException(Exception):
    pass

class NoActiveBlock(BaseTetrisException):
    pass

class CannotSpawnBlock(BaseTetrisException):
    pass

class CannotDropBlock(BaseTetrisException):
    pass

class CannotNudge(BaseTetrisException):
    pass