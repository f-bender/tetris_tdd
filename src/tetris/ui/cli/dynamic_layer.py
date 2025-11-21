"""Functions for creating numpy arrays that define the shape of the dynamic layer of the CLI UI."""

import inspect
import random
from typing import Literal, get_args, get_origin

import numpy as np
from numpy.typing import NDArray


def mix_layers(
    lhs: NDArray[np.uint32], rhs: NDArray[np.uint32], /, *, method: Literal["min", "max"] = "min"
) -> NDArray[np.uint32]:
    if lhs.shape != rhs.shape:
        msg = "Can't mix layers of differing shapes"
        raise ValueError(msg)

    return (np.minimum if method == "min" else np.maximum)(lhs, rhs)


# layer constructors


def cardinal_layer(
    size: tuple[int, int], direction: Literal["up", "down", "left", "right"] = "right"
) -> NDArray[np.uint32]:
    match direction:
        case "left":
            return np.tile(np.arange(size[1], dtype=np.uint32), (size[0], 1))
        case "right":
            return np.tile(np.arange(size[1], dtype=np.uint32)[::-1], (size[0], 1))
        case "up":
            return np.tile(np.arange(size[0], dtype=np.uint32)[:, np.newaxis], size[1])
        case "down":
            return np.tile(np.arange(size[0], dtype=np.uint32)[::-1, np.newaxis], size[1])


def diagonal_layer(
    size: tuple[int, int],
    y_direction: Literal["up", "down"] = "down",
    x_direction: Literal["left", "right"] = "right",
) -> NDArray[np.uint32]:
    y_range = np.arange(size[0], dtype=np.uint32)
    if y_direction == "down":
        y_range = y_range[::-1]

    x_range = np.arange(size[1], dtype=np.uint32)
    if x_direction == "right":
        x_range = x_range[::-1]

    return np.add.outer(y_range, x_range)


def circular_layer(
    size: tuple[int, int],
    direction: Literal["inward", "outward"] = "outward",
    y_center: Literal["top", "center", "bottom", "random"] = "random",
    x_center: Literal["left", "center", "right", "random"] = "random",
) -> NDArray[np.uint32]:
    match y_center:
        case "top":
            y_center_idx = 0
        case "center":
            y_center_idx = size[0] // 2
        case "bottom":
            y_center_idx = size[0]
        case "random":
            y_center_idx = random.randrange(size[0])

    match x_center:
        case "left":
            x_center_idx = 0
        case "center":
            x_center_idx = size[1] // 2
        case "right":
            x_center_idx = size[1]
        case "random":
            x_center_idx = random.randrange(size[1])

    x_coordinate_array = np.tile(np.arange(size[1]), (size[0], 1))
    y_coordinate_array = np.tile(np.arange(size[0])[:, np.newaxis], size[1])

    distance_array = np.sqrt(
        (x_coordinate_array - x_center_idx) ** 2 + (y_coordinate_array - y_center_idx) ** 2
    ).astype(np.uint32)
    if direction == "outward":
        distance_array = np.max(distance_array) - distance_array

    return distance_array


def random_layer(size: tuple[int, int], *, n_mixed_primitive_layers: int = 1) -> NDArray[np.uint32]:
    if n_mixed_primitive_layers < 1:
        msg = "n_mixed_primitive_layers must at least be 1"
        raise ValueError(msg)

    dynamic_layer: NDArray[np.uint32] | None = None
    for _ in range(n_mixed_primitive_layers):
        constructor = random.choice([cardinal_layer, diagonal_layer, circular_layer])

        parameters = inspect.signature(constructor).parameters  # type: ignore[arg-type]

        kwargs = {}
        for name, param in parameters.items():
            if name == "size":
                continue

            assert get_origin(param.annotation) is Literal
            kwargs[name] = random.choice(get_args(param.annotation))

        new_dynamic_layer = constructor(size=size, **kwargs)  # type: ignore[operator]

        if dynamic_layer is None:
            dynamic_layer = new_dynamic_layer
        else:
            dynamic_layer = mix_layers(dynamic_layer, new_dynamic_layer, method=random.choice(["min", "max"]))

    assert dynamic_layer is not None
    return dynamic_layer
