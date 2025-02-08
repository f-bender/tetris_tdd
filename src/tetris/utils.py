from typing import Any


def deep_merge(d1: dict[Any, Any], d2: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merges two dictionaries."""
    merged = d1.copy()
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
