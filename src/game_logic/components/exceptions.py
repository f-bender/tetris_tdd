from exceptions import BaseTetrisException


class NoActiveBlock(BaseTetrisException):
    pass


class CannotSpawnBlock(BaseTetrisException):
    pass


class CannotDropBlock(BaseTetrisException):
    pass


class CannotNudge(BaseTetrisException):
    pass

class ActiveBlockOverlap(BaseTetrisException):
    pass