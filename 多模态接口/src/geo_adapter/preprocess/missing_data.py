from __future__ import annotations

import numpy as np


def max_consecutive_false(mask: np.ndarray) -> int:
    """Return the longest invalid run in a one-dimensional Boolean mask."""
    best = current = 0
    for valid in np.asarray(mask, dtype=bool):
        if valid:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def interpolate_short_gaps(
    values: np.ndarray, max_gap_samples: int = 3, enabled: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate only bounded short NaN/Inf runs.

    Returns cleaned values, the original valid mask, and an interpolation mask.
    Leading/trailing gaps and runs longer than ``max_gap_samples`` are retained.
    """
    original = np.asarray(values, dtype=float)
    cleaned = original.copy()
    valid = np.isfinite(original)
    interpolated = np.zeros(original.shape, dtype=bool)
    if not enabled or max_gap_samples <= 0 or valid.sum() < 2:
        return cleaned, valid, interpolated

    index = 0
    size = len(original)
    while index < size:
        if valid[index]:
            index += 1
            continue
        start = index
        while index < size and not valid[index]:
            index += 1
        end = index
        gap = end - start
        if start > 0 and end < size and gap <= max_gap_samples:
            cleaned[start:end] = np.interp(
                np.arange(start, end), [start - 1, end], [cleaned[start - 1], cleaned[end]]
            )
            interpolated[start:end] = True
    return cleaned, valid, interpolated

