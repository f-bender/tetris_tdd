import contextlib
import logging
import random
from typing import Any

import numpy as np
from colorcet import cm as colorcet_colormaps
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)


_ADJACENT_DISTANCE_THRESHOLD = 0.05


def _is_cyclic(cmap: Any) -> bool:  # noqa: ANN401
    """Detect cyclic maps via color wraparound OR name heuristic."""
    name = cmap.name.lower()

    # Fast heuristic: name contains 'cyc'
    if "cyc" in name:
        return True

    # Deep heuristic: first and last colors are similar enough
    with contextlib.suppress(Exception):
        colors = cmap(np.linspace(0, 1, 256))[:, :3]
        return bool(np.linalg.norm(colors[0] - colors[-1]) < _ADJACENT_DISTANCE_THRESHOLD)

    return False


_CYCLIC_COLORCET_COLORMAPS: dict[str, Any] = {
    name: cmap for name, cmap in colorcet_colormaps.items() if cmap is not None and _is_cyclic(cmap)
}
assert _CYCLIC_COLORCET_COLORMAPS, (
    "No cyclic colorcet colormaps found - this might happen because matplotlib is not installed"
)


def get_colorcet_colormap(length: int = 256, name: str | None = None) -> NDArray[np.float64]:
    """Get colorcet colormap as a float numpy array (range 0-1) of RGB values.

    Args:
        length: Length of the returned colormap (i.e. shape[0])
        name: Name of the colorcet colormap to return; None for a random one.

    Returns:
        Numpy float array of shape [length, 3] of range 0-1.
    """
    name = name or random.choice(list(_CYCLIC_COLORCET_COLORMAPS))

    colormap = colorcet_colormaps.get(name)
    if colormap is None:
        msg = f"'{name}' is not a valid cyclic colorcet colormap"
        raise ValueError(msg)

    _LOGGER.debug("Got colorcet colormap '%s'", name)

    return colormap(np.linspace(0, 1, length))[:, :3]
