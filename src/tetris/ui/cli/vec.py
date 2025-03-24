from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, slots=True)
class Vec:
    y: int
    x: int

    def __add__(self, other: Self) -> "Vec":
        return Vec(self.y + other.y, self.x + other.x)

    def __sub__(self, other: Self) -> "Vec":
        return Vec(self.y - other.y, self.x - other.x)
