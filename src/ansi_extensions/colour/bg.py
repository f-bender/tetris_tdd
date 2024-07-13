def rgb_palette(r: int, g: int, b: int) -> str:
    return f"\x1b[48;5;{16 + int(r / 256 * 6) * 36 + int(g / 256 * 6) * 6 + int(b / 256 * 6)}m"


def rgb_truecolor(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"
